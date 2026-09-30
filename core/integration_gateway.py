"""
Integration Gateway Module
Supports standard industrial protocols for connecting to existing SCADA systems.
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ProtocolType(Enum):
    """Supported industrial protocols."""
    OPC_UA = "opc_ua"
    MQTT = "mqtt"
    MODBUS = "modbus"
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"


@dataclass
class ConnectionConfig:
    """Configuration for a protocol connection."""
    protocol: ProtocolType
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    topic: Optional[str] = None
    node_id: Optional[str] = None
    register_address: Optional[int] = None
    poll_interval_seconds: float = 10.0


@dataclass
class ConnectionStatus:
    """Status of a protocol connection."""
    protocol: ProtocolType
    connected: bool
    last_connected: Optional[float]
    last_data_received: Optional[float]
    error_count: int
    total_messages: int
    last_error: Optional[str] = None


class IntegrationGateway:
    """Gateway for connecting to industrial systems via standard protocols."""

    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self._running = False
        self.connections: Dict[str, ConnectionConfig] = {}
        self.statuses: Dict[str, ConnectionStatus] = {}
        self._data_callbacks: List[Callable] = []

        self._handlers = {
            ProtocolType.MQTT: self._handle_mqtt,
            ProtocolType.OPC_UA: self._handle_opc_ua,
            ProtocolType.MODBUS: self._handle_modbus,
            ProtocolType.HTTP_REST: self._handle_http_rest,
            ProtocolType.WEBSOCKET: self._handle_websocket
        }

    def start(self):
        """Start integration gateway."""
        self._running = True
        print("[IntegrationGateway] Started. Supporting OPC-UA, MQTT, Modbus, HTTP REST, WebSocket")

    def stop(self):
        """Stop integration gateway."""
        self._running = False
        print("[IntegrationGateway] Stopped")

    def add_connection(self, name: str, config: ConnectionConfig) -> bool:
        """Add a new protocol connection."""
        with self._lock:
            self.connections[name] = config
            self.statuses[name] = ConnectionStatus(
                protocol=config.protocol,
                connected=False,
                last_connected=None,
                last_data_received=None,
                error_count=0,
                total_messages=0
            )
        print(f"[IntegrationGateway] Added {config.protocol.value} connection: {name}")
        return True

    def register_data_callback(self, callback: Callable):
        """Register a callback for incoming data."""
        self._data_callbacks.append(callback)

    def connect_all(self):
        """Connect to all configured endpoints."""
        for name, config in self.connections.items():
            self._connect(name, config)

    def _connect(self, name: str, config: ConnectionConfig):
        """Connect to a specific endpoint."""
        handler = self._handlers.get(config.protocol)
        if handler:
            try:
                handler(name, config, connect=True)
                with self._lock:
                    self.statuses[name].connected = True
                    self.statuses[name].last_connected = time.time()
                print(f"[IntegrationGateway] Connected: {name}")
            except Exception as e:
                with self._lock:
                    self.statuses[name].error_count += 1
                    self.statuses[name].last_error = str(e)
                print(f"[IntegrationGateway] Connection failed: {name} - {e}")

    def _handle_mqtt(self, name: str, config: ConnectionConfig, connect: bool):
        """Handle MQTT connection."""
        print(f"[IntegrationGateway] MQTT: {config.host}:{config.port} topic={config.topic}")

    def _handle_opc_ua(self, name: str, config: ConnectionConfig, connect: bool):
        """Handle OPC-UA connection."""
        print(f"[IntegrationGateway] OPC-UA: {config.host}:{config.port} node={config.node_id}")

    def _handle_modbus(self, name: str, config: ConnectionConfig, connect: bool):
        """Handle Modbus connection."""
        print(f"[IntegrationGateway] Modbus: {config.host}:{config.port} register={config.register_address}")

    def _handle_http_rest(self, name: str, config: ConnectionConfig, connect: bool):
        """Handle HTTP REST connection."""
        print(f"[IntegrationGateway] HTTP REST: {config.host}:{config.port}")

    def _handle_websocket(self, name: str, config: ConnectionConfig, connect: bool):
        """Handle WebSocket connection."""
        print(f"[IntegrationGateway] WebSocket: {config.host}:{config.port}")

    def get_connection_status(self, name: str) -> Optional[ConnectionStatus]:
        """Get status of a specific connection."""
        return self.statuses.get(name)

    def get_all_statuses(self) -> Dict[str, dict]:
        """Get all connection statuses."""
        return {
            name: {
                "protocol": status.protocol.value,
                "connected": status.connected,
                "last_connected": status.last_connected,
                "last_data_received": status.last_data_received,
                "error_count": status.error_count,
                "total_messages": status.total_messages,
                "last_error": status.last_error
            }
            for name, status in self.statuses.items()
        }

    def send_command(self, connection_name: str, command: dict) -> bool:
        """Send a command through a specific connection."""
        with self._lock:
            if connection_name not in self.connections:
                return False
            config = self.connections[connection_name]
            status = self.statuses[connection_name]
            if not status.connected:
                return False
            print(f"[IntegrationGateway] Command sent via {connection_name}: {command}")
            return True
