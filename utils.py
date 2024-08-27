class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all the statistics."""
        self.val = 0.0     # Current value
        self.avg = 0.0     # Average value
        self.sum = 0.0     # Sum of all values
        self.count = 0     # Number of updates

    def update(self, val, n=1):
        """
        Updates the meter with a new value.
        
        Args:
            val (float): The new value to add.
            n (int): The number of occurrences of this value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count