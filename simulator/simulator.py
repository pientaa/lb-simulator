import pandas as pd

num_of_parallel_requests = 5
period = 5.0


class RequestQueue:
    def __init__(self):
        self.items = pd.DataFrame(columns=["id", "timestamp", "shard", "load", "expected_end_time", "actual_end_time"])

    def isEmpty(self):
        return self.items.empty

    def produce(self, item):
        self.items = self.items.append(item)
        self.items = self.items.sort_values(by=["timestamp"])

    def consume(self):
        consumed = self.items.head(1)
        self.items.drop(self.items.head(1).index, inplace=True)
        return consumed

    def size(self):
        return len(self.items)


class ProcessingQueue(RequestQueue):
    def removeFirst(self):
        """Removes only the first request and returns this request"""
        candidate = self.items.head(1)

        candidate = candidate.assign(actual_end_time=candidate["timestamp"].item() + candidate["load"].item())
        self.items.drop(candidate["id"], inplace=True)
        return candidate

    def removeFirstOutdated(self, current_timestamp):
        """Removes only the first outdated request up to current timestamp and returns this request"""
        candidate = self.items.head(1)

        if (candidate["timestamp"].item() + candidate["load"].item()) <= current_timestamp:
            candidate = candidate.assign(actual_end_time=candidate["timestamp"].item() + candidate["load"].item())
            self.items.drop(candidate["id"], inplace=True)
            return candidate
        else:
            return pd.DataFrame(columns=["id", "timestamp", "shard", "load", "expected_end_time", "actual_end_time"])

    def removeAllOutdated(self, current_timestamp):
        """Removes all outdated requests up to current timestamp and returns removed requests"""

        items_removed = pd.DataFrame(
            columns=["id", "timestamp", "shard", "load", "expected_end_time", "actual_end_time"])

        for index, item in self.items.iterrows():
            if (item["timestamp"] + item["load"]) <= current_timestamp:
                item["actual_end_time"] = item["timestamp"] + item["load"]
                self.items.drop(item["id"], inplace=True)
                items_removed = items_removed.append(item)
        return items_removed


def simulator(parallel_requests, new_period):
    global num_of_parallel_requests, period
    num_of_parallel_requests = parallel_requests
    period = new_period

    print("Simulation started with following params:")
    print("{ parallel_requests: " + str(num_of_parallel_requests) + ", period: " + str(period) + " } \n")

    requests = pd.read_csv("./generator/requests.csv")

    requests = requests.assign(
        expected_end_time=lambda dataframe: dataframe["timestamp"] + dataframe["load"],
        actual_end_time=None
    )
    requests = requests.rename(columns={"Unnamed: 0": "id"}, inplace=False)

    shards_on_nodes = pd.read_csv("./simulator/shard_allocated.csv")

    requests_completed = RequestQueue()

    for (node, shards) in shards_on_nodes.groupby('node'):
        requests_processing = ProcessingQueue()
        requests_awaiting = RequestQueue()

        requests_per_node = requests[requests["shard"].isin(shards["shard"].to_list())]

        last_completed_request_timestamp = 0.0

        for index, request in requests_per_node.iterrows():
            current_timestamp = request["timestamp"]

            # What if processing is not full and some are awaiting?
            while requests_processing.size() < num_of_parallel_requests and requests_awaiting.size() > 0:
                awaiting_request = requests_awaiting.consume()
                time_expected_to_start = awaiting_request['timestamp'].item()

                if last_completed_request_timestamp > time_expected_to_start:
                    awaiting_request = awaiting_request.assign(timestamp=last_completed_request_timestamp)

                requests_processing.produce(awaiting_request)

            while requests_awaiting.size() > 0:
                processed_request = requests_processing.removeFirstOutdated(current_timestamp)

                if not processed_request.empty:
                    requests_completed.produce(processed_request)
                    last_completed_request_timestamp = processed_request["actual_end_time"].item()
                    awaiting_request = requests_awaiting.consume()

                    awaiting_request = awaiting_request.assign(timestamp=processed_request["actual_end_time"].item())

                    requests_processing.produce(awaiting_request)
                else:
                    break

            # If there is any awaiting, this should return empty dataframe
            removed_requests = requests_processing.removeAllOutdated(current_timestamp)

            if not removed_requests.empty:
                last_completed_request_timestamp = removed_requests.sort_values(by=['actual_end_time']).tail(1)['actual_end_time'].item()

            if not removed_requests.empty:
                for index, req in removed_requests.iterrows():
                    requests_completed.produce(req)

            if requests_processing.size() < num_of_parallel_requests:
                requests_processing.produce(request)
            else:
                requests_awaiting.produce(request)

        # It's impossible that there is any awaiting if processing is not full
        # while requests_processing.size() > 0 or requests_awaiting.size() > 0:
        while requests_processing.size() > 0:
            removed = requests_processing.removeFirst()
            requests_completed.produce(removed)

            if not requests_awaiting.isEmpty():
                awaiting_request = requests_awaiting.consume()
                awaiting_request = awaiting_request.assign(timestamp=removed["actual_end_time"].item())
                requests_processing.produce(awaiting_request)

    requests_completed_df = requests_completed.items.sort_values(by=["timestamp"]).assign(
        delay=lambda dataframe: dataframe["actual_end_time"] - dataframe["expected_end_time"]
    ).round(3)

    return requests_completed_df


if __name__ == "__main__":
    simulator()
