# SIMPLE HOLD-OUT VALIDATION

```python
num_validation_samples = 10000 
np.random.shuffle(data)
validation_data = data[:num_validation_samples] 
data = data[num_validation_samples:] 
training_data = data[:]
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data) # At this point you can tune your model,
# retrain it, evaluate it, tune it again...
model = get_model() 
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)
```

### K-Fold Validation

```python
k=4
num_validation_samples = len(data) // k 
np.random.shuffle(data) 
validation_scores = []

for fold in range(k):
    validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data) 
    validation_scores.append(validation_score)
    validation_score = np.average(validation_scores) 
    model = get_model()
    model.train(data)
    test_score = model.evaluate(test_data)
```

### ITERATED K-FOLD VALIDATION WITH SHUFFLING