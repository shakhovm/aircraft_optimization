
class UpdatingModel:
    def update_weights(self, params):
        for target_weight, weight in zip(self.parameters(), params):
            target_weight.data.copy_(target_weight.data + self.tau * (weight.data - target_weight.data))