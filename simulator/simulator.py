import numpy as np
import pandas as pd
import sys

class RequestQueue:
    def __init__(self):
        self.items = pd.DataFrame(columns=["id", "timestamp", "shard", "load", "expected_end_time", "actual_end_time"])

    def isEmpty(self):
        return self.items.empty

    def produce(self, item):
        self.items = self.items.append(item)
        self.items = self.items.sort_values(by=["expected_end_time"])
        
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

        candidate = candidate.assign(actual_end_time = candidate["timestamp"].item() + candidate["load"].item())
        self.items.drop(candidate["id"], inplace=True)
        return candidate


    def removeFirstOutdated(self, current_timestamp):
        """Removes only the first outdated request up to current timestamp and returns this request"""
        candidate = self.items.head(1)

        if((candidate["timestamp"].item() + candidate["load"].item()) < current_timestamp):
            candidate = candidate.assign(actual_end_time = candidate["timestamp"].item() + candidate["load"].item())
            self.items.drop(candidate["id"], inplace=True)
            return candidate
        else:
            return pd.DataFrame(columns=["id", "timestamp", "shard", "load", "expected_end_time", "actual_end_time"])   

    def removeAllOutdated(self, current_timestamp):
        """Removes all outdated requests up to current timestamp and returns removed requests"""

        items_removed = pd.DataFrame(columns=["id", "timestamp", "shard", "load", "expected_end_time", "actual_end_time"])

        for index, item in self.items.iterrows():
            if((item["timestamp"] + item["load"]) < current_timestamp):
                item["actual_end_time"] = item["timestamp"] + item["load"]
                self.items.drop(item["id"], inplace=True)
                items_removed = items_removed.append(item)
        return items_removed


def simulator():
    requests = pd.read_csv("./generator/requests.csv")
    
    requests = requests.assign(
        expected_end_time = lambda dataframe: dataframe["timestamp"] + dataframe["load"],
        actual_end_time = None
    )
    requests = requests.rename(columns={"Unnamed: 0": "id"}, inplace = False)

    shards_on_nodes = pd.read_csv("./simulator/shard_allocated.csv")

    requests_processed = ProcessingQueue()
    requests_awaiting = RequestQueue()
    requests_completed = RequestQueue()

    for (node, shards) in shards_on_nodes.groupby('node'):

        requests_per_node = requests[requests["shard"].isin(shards["shard"].to_list())]

        for index, request in requests_per_node.iterrows():
            current_timestamp = request["timestamp"]

            while(requests_awaiting.size() > 0):
                removed_request = requests_processed.removeFirstOutdated(current_timestamp)

                if (not removed_request.empty):
                    requests_completed.produce(removed_request)
                    awaiting_request = requests_awaiting.consume()

                    awaiting_request = awaiting_request.assign(timestamp = removed_request["actual_end_time"].item())

                    requests_processed.produce(awaiting_request)
                else:
                    break
            
            removed_requests = requests_processed.removeAllOutdated(current_timestamp)

            if(not removed_requests.empty):
                for index, req in removed_requests.iterrows():
                    requests_completed.produce(req)
      
            if(requests_processed.size() < num_of_parallel_requests):
                requests_processed.produce(request)
            else:
                requests_awaiting.produce(request)

        # TODO: CONSUME AWAITING REQUESTS
        while(requests_processed.size() > 0 or requests_awaiting.size() > 0):
            removed_request = requests_processed.removeFirst()
            
            requests_completed.produce(removed_request)

            if(not requests_awaiting.isEmpty()):
                awaiting_request = requests_awaiting.consume()

                awaiting_request = awaiting_request.assign(timestamp = removed_request["actual_end_time"].item())

                requests_processed.produce(awaiting_request)


    requests_completed_df = requests_completed.items.sort_values(by=["timestamp"]).assign(
        delay = lambda dataframe: dataframe["actual_end_time"] - dataframe["expected_end_time"]
    ).round(3)

    requests_completed_df.to_csv('./simulator/requests_completed.csv', index=False) 


if __name__ == "__main__":
    num_of_parallel_requests = int(sys.argv[1])
    period = float(sys.argv[2])
    simulator()