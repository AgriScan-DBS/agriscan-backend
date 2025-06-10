import fetch from "node-fetch";
import FormData from "form-data";
import fs from "fs";
import path from "path";

async function testPredict() {
  try {
    // First, login to get the token
    const loginResponse = await fetch("http://localhost:5000/api/auth/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        userName: "rafliafrz_",
        password: "Rafli12345!",
      }),
    });

    const loginData = await loginResponse.json();
    if (!loginResponse.ok) {
      throw new Error(`Login failed: ${loginData.message}`);
    }

    const token = loginData.data.token;
    console.log("✅ Login successful");

    // Create form data with image
    const form = new FormData();
    const imagePath = path.join(process.cwd(), "test-image.png"); // Put a test image in your project root
    form.append("image", fs.createReadStream(imagePath));

    // Make prediction request
    console.log("🔄 Sending prediction request...");
    const predictResponse = await fetch("http://localhost:5000/api/predict", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        ...form.getHeaders(),
      },
      body: form,
    });

    const predictData = await predictResponse.json();
    console.log("Response status:", predictResponse.status);
    console.log("Response headers:", predictResponse.headers);
    console.log("Response body:", JSON.stringify(predictData, null, 2));
  } catch (error) {
    console.error("❌ Test failed:", error);
  }
}

testPredict();
