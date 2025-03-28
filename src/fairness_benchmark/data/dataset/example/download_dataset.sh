#!/usr/bin/env bash

echo "Downloading Adult Dataset ..."
curl -sSL "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" >"adult.data"
echo "Data downloaded..."
curl -sSL "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" >"adult.test"
echo "Test downloaded..."
curl -sSL "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names" >"adult.names"
echo "Names downloaded..."
echo "Downloaded Adult Dataset."

echo "Downloading German Dataset as an example"
curl -sSL "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data" >"example.data"
curl -sSL "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc" >"example.doc"
echo "Downloaded example."
