import Application from "./app.js";

const app = new Application();

// Handle unhandled promise rejections
process.on("unhandledRejection", (err) => {
  console.log("Unhandled Rejection:", err);
  process.exit(1);
});

// Start the server
app.startServer();
