import argparse
import csv
import io
import json

import requests
from kafka import KafkaProducer

API_BASE_URL = "http://localhost:8000"


def build_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=20,
    )


def stream_endpoint_to_topic(endpoint: str, topic: str, bootstrap_servers: str) -> None:
    producer = build_producer(bootstrap_servers)
    url = f"{API_BASE_URL}/stream/{endpoint}"
    print(f"Connecting to {url}, publishing to topic '{topic}' ({bootstrap_servers})")

    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        header = None
        sent = 0
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            reader = csv.reader(io.StringIO(raw_line))
            row = next(reader)
            if header is None:
                header = row
                continue
            record = dict(zip(header, row))
            producer.send(topic, value=record)
            sent += 1
            if sent % 200 == 0:
                print(f"  sent {sent} messages to '{topic}'")

    producer.flush()
    print(f"Stream ended. Total messages sent: {sent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, choices=["car_data", "location", "intervals", "weather", "full"])
    parser.add_argument("--topic", required=True)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    args = parser.parse_args()

    stream_endpoint_to_topic(args.endpoint, args.topic, args.bootstrap_servers)
