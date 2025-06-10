import ModelService from "../services/ModelService.js";
import path from "path";
import fs from "fs";

class InferenceController {
  predict = async (req, h) => {
    try {
      console.log("📝 Request received for prediction");

      // Check if request payload exists
      if (!req.payload) {
        console.error("❌ No payload received");
        return h
          .response({
            status: 400,
            message: "No payload received",
          })
          .code(400);
      }

      // Log the payload structure
      console.log("📦 Payload structure:", Object.keys(req.payload));

      // Check for image file
      const image = req.payload.image;
      if (!image) {
        console.error("❌ No image file in payload");
        return h
          .response({
            status: 400,
            message: "No image file provided in the request",
          })
          .code(400);
      }

      // Log file details
      console.log("📸 Received image:", {
        filename: image.filename,
        headers: image.headers,
        path: image.path,
      });

      // Ensure the file was properly saved
      if (!fs.existsSync(image.path)) {
        console.error("❌ Image file not found at path:", image.path);
        return h
          .response({
            status: 500,
            message: "File upload failed - file not found",
          })
          .code(500);
      }

      try {
        // Perform prediction
        console.log("🔄 Starting prediction process");
        const results = await ModelService.predict(image.path);

        // Clean up: remove temporary file
        try {
          fs.unlinkSync(image.path);
          console.log("🗑️ Temporary file cleaned up");
        } catch (cleanupError) {
          console.warn("⚠️ Failed to clean up temporary file:", cleanupError);
        }

        console.log("✅ Prediction completed successfully");
        return h
          .response({
            status: 200,
            data: results,
            message: "Prediction successful",
          })
          .code(200);
      } catch (predictionError) {
        console.error("❌ Prediction error:", predictionError);

        // Clean up on error
        try {
          if (fs.existsSync(image.path)) {
            fs.unlinkSync(image.path);
            console.log("🗑️ Cleaned up file after error");
          }
        } catch (cleanupError) {
          console.warn("⚠️ Failed to clean up file after error:", cleanupError);
        }

        return h
          .response({
            status: 500,
            message: `Prediction failed: ${predictionError.message}`,
          })
          .code(500);
      }
    } catch (error) {
      console.error("❌ Controller error:", error);
      return h
        .response({
          status: 500,
          message: `Server error: ${error.message}`,
        })
        .code(500);
    }
  };
}

export default new InferenceController();
