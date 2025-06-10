import mongoose from "mongoose";
import { configDotenv } from "dotenv";

configDotenv();

const clientOptions = {
  serverApi: {
    version: "1",
    strict: true,
    deprecationErrors: true,
  },
  // Opsi khusus untuk MongoDB Atlas
  retryWrites: true,
  w: "majority",
  maxPoolSize: 10, // Maintain up to 10 socket connections
  serverSelectionTimeoutMS: 5000, // Keep trying to send operations for 5 seconds
  socketTimeoutMS: 45000, // Close sockets after 45 seconds of inactivity
  bufferCommands: false, // Disable mongoose buffering
};

let isConnected = false;

const connectDB = async () => {
  if (isConnected) {
    console.log("Using existing database connection");
    return;
  }

  try {
    // Validasi environment variable
    if (!process.env.MONGO_URI) {
      throw new Error(
        "MONGO_URI environment variable is not defined. Please check your .env file"
      );
    }

    // Validasi format MongoDB Atlas URI
    if (!process.env.MONGO_URI.includes("mongodb+srv://")) {
      console.warn(
        "⚠️  Warning: URI doesn't appear to be a MongoDB Atlas connection string"
      );
    }

    console.log("🔄 Connecting to MongoDB Atlas...");

    const conn = await mongoose.connect(process.env.MONGO_URI, clientOptions);

    isConnected = true;
    console.log(`✅ MongoDB Atlas Connected Successfully!`);
  } catch (error) {
    console.error("❌ Failed to connect to MongoDB Atlas:", error);

    // Specific error handling untuk MongoDB Atlas
    if (error instanceof Error) {
      if (error.message.includes("authentication failed")) {
        console.error(
          "🔐 Authentication Error: Check your username/password in the connection string"
        );
      } else if (error.message.includes("getaddrinfo ENOTFOUND")) {
        console.error(
          "🌐 Network Error: Check your internet connection and cluster URL"
        );
      } else if (error.message.includes("IP not whitelisted")) {
        console.error(
          "🛡️  IP Whitelist Error: Add your IP to MongoDB Atlas Network Access"
        );
      }

      console.error("Error details:", error.message);
    }

    process.exit(1);
  }
};

export const disconnectDB = async () => {
  if (!isConnected) {
    return;
  }

  try {
    await mongoose.disconnect();
    isConnected = false;
    console.log("MongoDB disconnected");
  } catch (error) {
    console.error("Error disconnecting from MongoDB:", error);
    process.exit(1);
  }
};

export default connectDB;
