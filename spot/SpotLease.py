from bosdyn.client.lease import ResourceAlreadyClaimedError, LeaseKeepAlive


class SpotLease:
    def __init__(self, client):
        self.lease_client = client
        self.lease = None
        self.lease_keepalive = None

    def toggle_lease(self):
        if self.lease_keepalive is None:
            return self.start_lease()
        else:
            return self.return_lease()

    def start_lease(self):
        # check lease
        try:
            print("Lease Acquire")
            self.lease = self.lease_client.acquire()
        except ResourceAlreadyClaimedError as err:
            print("The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds.")
            self.lease = self.lease_client.take()
            # return

        self.lease_keepalive = LeaseKeepAlive(self.lease_client, on_failure_callback=self.return_lease)
        return True

    def return_lease(self):
        try:
            self.lease_keepalive.shutdown()
            # self.lease_client.return_lease(self.lease)
        except RuntimeError:
            print("다른 기기에서 Lease를 가져갔습니다.")

        self.lease_keepalive = None
        self.lease = None

        return True
