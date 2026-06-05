import torch

class CUDAPrefetcher:
    def __init__(self, loader, device):
        self._orig_loader = loader  # Store the original loader for __len__
        self.device = device
        self.stream = torch.cuda.Stream()
        self.reset()

    def reset(self):
        """
        Reset the internal iterator and preload the first batch. Call this at the start of each epoch to reuse the prefetcher.
        """
        self.loader = iter(self._orig_loader)
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, (list, tuple)):
                self.next_data = [d.to(self.device, non_blocking=True) for d in self.next_data]
            elif isinstance(self.next_data, dict):
                self.next_data = {k: v.to(self.device, non_blocking=True) for k, v in self.next_data.items()}
            else:
                self.next_data = self.next_data.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is None:
            raise StopIteration
        data = self.next_data
        self.preload()
        return data

    def __len__(self):
        return len(self._orig_loader) 