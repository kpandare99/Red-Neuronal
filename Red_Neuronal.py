import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt #visualizar graficamente el aprendizaje del modelo


# Crear una matriz de datos de ejemplo
data = np.array([[0, 0, 0, 0, 0, 0],              
                 [0, 0, 0, 0, 0, 1],                   
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 1],
                 [0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1],
                 [0, 0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1, 1],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 1],
                 [0, 0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 1, 1],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 1],
                 [0, 1, 0, 0, 1, 0],
                 [0, 1, 0, 0, 1, 1],
                 [0, 1, 0, 1, 0, 0],
                 [0, 1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 1, 0],
                 [0, 1, 0, 1, 1, 1],
                 [0, 1, 1, 0, 0, 0],
                 [0, 1, 1, 0, 0, 1],
                 [0, 1, 1, 0, 1, 0],
                 [0, 1, 1, 0, 1, 1],
                 [0, 1, 1, 1, 0, 0],
                 [0, 1, 1, 1, 0, 1],
                 [0, 1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1],
                 [1, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 1, 1],
                 [1, 0, 0, 1, 0, 0],
                 [1, 0, 0, 1, 0, 1],
                 [1, 0, 0, 1, 1, 0],
                 [1, 0, 0, 1, 1, 1],
                 [1, 0, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 1],
                 [1, 0, 1, 0, 1, 0],
                 [1, 0, 1, 0, 1, 1],
                 [1, 0, 1, 1, 0, 0],
                 [1, 0, 1, 1, 0, 1],
                 [1, 0, 1, 1, 1, 0],
                 [1, 0, 1, 1, 1, 1],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 1],
                 [1, 1, 0, 0, 1, 0],
                 [1, 1, 0, 0, 1, 1],
                 [1, 1, 0, 1, 0, 0],
                 [1, 1, 0, 1, 0, 1],
                 [1, 1, 0, 1, 1, 0],
                 [1, 1, 0, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 1],
                 [1, 1, 1, 0, 1, 0],
                 [1, 1, 1, 0, 1, 1],
                 [1, 1, 1, 1, 0, 0],
                 [1, 1, 1, 1, 0, 1],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1]],dtype="float32"
)

target = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 0, 1,
                  0, 0, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 1,
                  0, 0, 0, 0, 1, 1, 0, 1,
                  1, 1, 0, 1, 1, 1, 0, 1,
                  1, 1, 0, 1, 1, 1, 0, 0,
                  1, 1, 0, 1, 1, 1, 0, 1],dtype="float32"
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=6, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["binary_accuracy"])

modelo = model.fit(data, target, epochs=1000)

scores = model.evaluate(data, target)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(data).round())

plt.xlabel("# Epoca")
plt.ylabel("Magnitud perdida")
plt.plot(modelo.history["loss"])
