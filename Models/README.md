### Instructions


load directly the repository using this code

Load model<br>
loaded_model = tf.keras.models.load_model('./best_model_simple_rnn')<br>
loaded_model.summary()<br>
assert np.allclose(model_simple_rnn.predict(X_test), loaded_model.predict(X_test))<br>
