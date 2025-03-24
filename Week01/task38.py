import numpy as np
    
class Xor:
    def __init__(self, learning_rate=1e-1, epochs=100_000, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.bias_or = -1
        self.bias_and = -6
        self.bias_nand = 7
        self.w1_or, self.w2_or = self.train_or()
        self.w1_and, self.w2_and = self.train_and()
        self.w1_nand, self.w2_nand = self.train_nand()

    def _create_or_dataset(self):
        return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    def _create_and_dataset(self):
        return np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

    def _create_nand_dataset(self):
        return np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    
    def _create_xor_dataset(self):
        return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _initialize_weights(self, x, y):
        return np.random.uniform(x, y, size=2)

    def _calculate_loss(self, w1, w2, bias, dataset):
        loss = 0
        for (x, y, target) in dataset:
            prediction_sigmoid = self._sigmoid(w1 * x + w2 * y + bias)
            loss += (prediction_sigmoid - target) ** 2
        loss /= len(dataset)
        return loss
        
    def _experiment(self, bias, dataset):
        w1, w2 = self._initialize_weights(-1, 1)

        for epoch in range(self.epochs):
            loss = self._calculate_loss(w1, w2, bias, dataset)
            loss_plus_epsilon = self._calculate_loss(w1 + self.epsilon, w2 + self.epsilon, bias, dataset)
            L = (loss_plus_epsilon - loss) / self.epsilon

            offset = self.learning_rate * L
            w1 -= offset
            w2 -= offset

            # if epoch % 1000 == 0:
            #     print(f"Epoch: {epoch}, Loss: {loss}, Weights: {w1, w2}")

        return w1, w2

    def train_or(self):
        dataset = self._create_or_dataset()
        return self._experiment(dataset=dataset, bias=self.bias_or)

    def train_and(self):
        dataset = self._create_and_dataset()
        return self._experiment(dataset=dataset, bias=self.bias_and)

    def train_nand(self):
        dataset = self._create_nand_dataset()
        return self._experiment(dataset=dataset, bias=self.bias_nand)

    def predict(self, x, y):
        or_prediction_value = self._sigmoid(self.w1_or * x + self.w2_or * y + self.bias_or)
        # print(f"OR Prediction: {or_prediction_value}")
        nand_prediction_value = self._sigmoid(self.w1_nand * x + self.w2_nand * y + self.bias_nand)
        # print(f"NAND Prediction: {nand_prediction_value}")
        and_prediction_value = self._sigmoid(self.w1_and * or_prediction_value + self.w2_and * nand_prediction_value + self.bias_and) 
        prediction = and_prediction_value >= 0.5
        return prediction, and_prediction_value

    def evaluate(self):
        dataset = self._create_xor_dataset()
        for (x, y, target) in dataset:
            print("-----------------------------------")
            prediction, prediction_value = self.predict(x, y)
            print(f"Input: {x, y}, Expected: {target}, Predicted: {int(prediction)}, Confidence: {prediction_value}")
        print("-----------------------------------")

def forward(model, x, y):
    prediction, _ = model.predict(x, y)
    print(f"Input: {x, y}, Predicted: {int(prediction)}")

if __name__ == '__main__':
    xor_model = Xor()
    forward(xor_model, 0, 0)
    forward(xor_model, 0, 1)
    forward(xor_model, 1, 0)
    forward(xor_model, 1, 1)
    # xor_model.evaluate()