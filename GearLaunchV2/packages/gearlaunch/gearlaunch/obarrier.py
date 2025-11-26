#!/usr/bin/env python3
#####################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

import argparse
import asyncio
import socket


class SyncServer:
    def __init__(self, port: int, num_nodes: int, timeout=60):
        self._port = port
        self._num_nodes = num_nodes
        self._server = None
        self._connected_ranks = set()
        self._connections = set()
        # Are we ready to send the completed message to all connected ranks?
        self._ready_to_send = asyncio.Event()
        self._failed = False
        self._failure_reason = False
        self._status_interval = 10
        self._timeout = timeout

    async def status_print(self):
        all_ranks = {i for i in range(1, self._num_nodes)}
        while True:
            outstanding_workers = all_ranks - self._connected_ranks
            print(
                f'{len(self._connected_ranks)}/{len(all_ranks)} workers connected, waiting on ranks: {", ".join(str(x) for x in outstanding_workers)}'
            )
            await asyncio.sleep(self._status_interval)

    async def run(self):
        print(
            f"Starting server on port {self._port}, expecting {self._num_nodes - 1} nodes to connect."
        )
        self._server = await asyncio.start_server(
            self.handle_connection, host="0.0.0.0", port=self._port
        )
        print("Server started successfully.")
        async with self._server:
            loop = asyncio.get_event_loop()
            server_task = loop.create_task(self._server.serve_forever())
            loop.create_task(self.status_print())

            # Set a timeout for waiting for all connections or for something to go wrong
            try:
                print(f"Waiting for all nodes to connect (timeout: {self._timeout} seconds)...")
                await asyncio.wait_for(self._ready_to_send.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                self.fail(
                    f"Timeout waiting for all nodes to connect within {self._timeout} seconds."
                )

            # Send messages to all connected ranks
            for connection in self._connections:
                if self._failed:
                    print("Sending FAILED message to connection.")
                    connection.write(f"FAILED: {self._failure_reason}".encode("utf-8"))
                else:
                    print("Sending OK message to connection.")
                    connection.write("OK".encode("utf-8"))
                connection.close()

            server_task.cancel()

        return not self._failed

    def fail(self, message):
        self._failed = True
        self._failure_reason = message
        self._ready_to_send.set()
        print(f"Failing due to: {message}")

    def add_rank(self, rank):
        print(f"New connection from rank {rank}")
        if rank in self._connected_ranks:
            self.fail(f"More than one node with rank {rank} connected!")
            return

        if rank < 1 or rank >= self._num_nodes:
            self.fail(
                f"Got connection from rank {rank} which is outside of the range [1, {self._num_nodes - 1}]"
            )

        self._connected_ranks.add(rank)
        print(f"Rank {rank} connected successfully. #Connected ranks: {len(self._connected_ranks)}")
        if len(self._connected_ranks) == self._num_nodes - 1:
            print("All nodes connected. Ready to send OK message.")
            self._ready_to_send.set()

    def remove_rank(self, rank):
        if rank is not None:
            self._connected_ranks.remove(rank)
            print(f"Rank {rank} disconnected. #Remaining ranks: {len(self._connected_ranks)}")

    async def handle_connection(self, reader, writer):
        try:
            rank = None
            # Store the connection in our list of connections
            self._connections.add(writer)
            print("New connection established. Waiting for rank info...")

            # The connection should send a single line, which is the rank
            line = await reader.readline()
            try:
                rank = int(line.decode("utf-8"))
            except ValueError as error:
                self.fail(f"Encountered exception {error} while decoding rank info.")
                return

            # Add this to the set of connected ranks
            self.add_rank(rank)

            # Wait for client to disconnect (Or to send extraneous data)
            await reader.read(1)
            self._connections.remove(writer)

        finally:
            if rank:
                self.remove_rank(rank)
            writer.close()
            await writer.wait_closed()
            print(f"Disconnecting rank {rank}")


async def run_client(host, port, rank):
    while True:
        try:
            print(f"Attempting to connect to server at {host}:{port} as rank {rank}...")
            reader, writer = await asyncio.open_connection(host, port)
            break
        except (ConnectionRefusedError, socket.gaierror) as error:
            print(f'Connection to rank 0 failed due to "{error}", trying again in 10s...')
            await asyncio.sleep(10)
    print(f"Successfully connected to server at {host}:{port}")
    writer.write(f"{rank}\n".encode("utf-8"))
    status = (await reader.read()).decode("utf-8")
    print(f"Received status from server: {status}")
    return status.startswith("OK")


def obarrier():
    # Parse and validate arguments
    parser = argparse.ArgumentParser(description="Allows multiple osmo tasks to synchronize")
    parser.add_argument(
        "--connect", help="Provide if this is not rank 0. The ip or hostname to connect to"
    )
    parser.add_argument("--port", type=int, default=12344, help="The port to connect to on rank 0")
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="A number from 0 to (n-1) where n is the number of nodes",
    )
    parser.add_argument("--num_nodes", type=int, required=True, help="The number of nodes")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 5,
        help="The number of seconds to wait for all nodes to connect",
    )

    args = parser.parse_args()

    if args.rank >= args.num_nodes:
        print(f"Rank ({args.rank}) must be less than num nodes ({args.num_nodes})")
        exit(1)

    if args.rank < 0:
        print(f"Rank ({args.rank}) must be greater than or equal to 0")
        exit(1)

    if args.num_nodes < 2:
        print("Number of nodes must be at least 2")
        exit(0)

    if not args.connect and args.rank != 0:
        print('Must provide "--connect <ip/hostname of rank 0>" flag when rank != 0')
        exit(1)

    if args.rank == 0:
        server = SyncServer(args.port, args.num_nodes, args.timeout)
        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(server.run())
    else:
        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(run_client(args.connect, args.port, args.rank))
    exit(0 if success else 1)


if __name__ == "__main__":
    obarrier()
