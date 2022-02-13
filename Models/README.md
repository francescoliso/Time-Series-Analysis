### Instructions


load directly the repository using this code

Load model\n
loaded_model = tf.keras.models.load_model('./best_model_simple_rnn')\n
loaded_model.summary()\n
assert np.allclose(model_simple_rnn.predict(X_test), loaded_model.predict(X_test))\n
