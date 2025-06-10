import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import fs from "fs";

class ModelService {
  constructor() {
    this.model = null;
    this.modelPath = path.join(process.cwd(), "tfjs_model");
    this.initModel();
  }

  async initModel() {
    try {
      const modelJsonPath = path.join(this.modelPath, "model.json");
      if (!fs.existsSync(modelJsonPath)) {
        throw new Error(
          "Model file not found. Please ensure model.json is in the tfjs_model directory."
        );
      }

      console.log("Loading TensorFlow.js model...");

      // Load model with custom loading handler
      this.model = await tf.loadGraphModel(`file://${modelJsonPath}`, {
        onProgress: (fraction) => {
          console.log(`Loading model: ${(fraction * 100).toFixed(1)}%`);
        },
      });

      // Warmup the model
      const dummyInput = tf.zeros([1, 224, 224, 3]);
      await this.model.predict(dummyInput).data();
      dummyInput.dispose();

      console.log("Model loaded successfully");
    } catch (error) {
      console.error("Error loading model:", error);
      throw new Error(`Failed to initialize model: ${error.message}`);
    }
  }

  async preprocessImage(imagePath) {
    try {
      // Read the image file
      const imageBuffer = fs.readFileSync(imagePath);

      // Decode image
      let tensor = tf.node.decodeImage(imageBuffer, 3);

      // Resize to model's expected size
      tensor = tf.image.resizeBilinear(tensor, [224, 224]);

      // Preprocess for ResNet50
      // Convert to float32 and normalize to [0, 1]
      tensor = tf.cast(tensor, "float32").div(255.0);

      // ResNet50 normalization
      const meanRGB = [0.485, 0.456, 0.406];
      const stdRGB = [0.229, 0.224, 0.225];

      // Normalize each channel
      const channels = tf.split(tensor, 3, 2);
      const normalizedChannels = channels.map((channel, i) =>
        channel.sub(meanRGB[i]).div(stdRGB[i])
      );
      tensor = tf.concat(normalizedChannels, 2);

      // Add batch dimension
      tensor = tensor.expandDims(0);

      return tensor;
    } catch (error) {
      console.error("Error preprocessing image:", error);
      throw new Error(`Failed to preprocess image: ${error.message}`);
    }
  }

  async predict(imagePath) {
    try {
      if (!this.model) {
        throw new Error("Model not initialized");
      }

      // Preprocess the image
      const tensor = await this.preprocessImage(imagePath);

      // Make prediction
      console.log("Running inference...");
      const predictions = await this.model.predict(tensor);

      // Get prediction data
      let results;
      if (Array.isArray(predictions)) {
        // If model returns multiple tensors, get the main prediction tensor
        results = await predictions[0].data();
        // Cleanup other tensors
        predictions.slice(1).forEach((tensor) => tensor.dispose());
      } else {
        // Single output tensor
        results = await predictions.data();
      }

      // Clean up tensors
      tensor.dispose();
      if (!Array.isArray(predictions)) {
        predictions.dispose();
      }

      // Process results
      const processedResults = this.processResults(Array.from(results));

      return processedResults;
    } catch (error) {
      console.error("Error during prediction:", error);
      throw new Error(`Failed to process image: ${error.message}`);
    }
  }

  processResults(predictions) {
    try {
      // Get class labels
      const labelPath = path.join(this.modelPath, "labels.json");
      let classLabels = ["class1", "class2", "class3"]; // Default labels

      if (fs.existsSync(labelPath)) {
        classLabels = JSON.parse(fs.readFileSync(labelPath, "utf8"));
      }

      // Get top 3 predictions
      const topK = 3;
      const topKIndices = predictions
        .map((prob, index) => ({ probability: prob, index: index }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, topK);

      return {
        topPredictions: topKIndices.map(({ probability, index }) => ({
          class: classLabels[index] || `class_${index}`,
          probability: probability,
          score: (probability * 100).toFixed(2) + "%",
        })),
      };
    } catch (error) {
      console.error("Error processing results:", error);
      throw new Error(`Failed to process results: ${error.message}`);
    }
  }
}

export default new ModelService();
