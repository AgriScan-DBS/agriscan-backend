import Hapi from "@hapi/hapi";
import AuthController from "./controllers/AuthController.js";
import InferenceController from "./controllers/InferenceController.js";
import { verifyToken } from "./middleware/auth.js";
import { uploadMiddleware } from "./middleware/upload.js";
import connectDB from "./config/database.js";

class Application {
  constructor() {
    this.server = this.initServer();
    this.initRoutes();
  }

  startServer = async () => {
    try {
      // Connect to database first
      await connectDB();

      // Then start the server
      await this.server.start();
      console.log(`Server running on ${this.server.info.uri}`);
    } catch (error) {
      console.error("Server start error:", error);
      process.exit(1);
    }
  };

  initServer = () => {
    return Hapi.server({
      port: process.env.PORT || 5000,
      host: process.env.HOST || "localhost",
      routes: {
        cors: {
          origin: ["*"],
        },
      },
    });
  };

  initRoutes = () => {
    // Health check route
    this.server.route({
      method: "GET",
      path: "/",
      handler: (request, h) => {
        return h
          .response({
            status: 200,
            data: "AgriScan API",
            message: "Server is running",
          })
          .code(200);
      },
    });

    this.authRoutes();
    this.inferenceRoutes();
  };

  authRoutes = () => {
    // Public routes
    this.server.route({
      method: "POST",
      path: "/api/auth/register",
      handler: AuthController.register,
    });

    this.server.route({
      method: "POST",
      path: "/api/auth/login",
      handler: AuthController.login,
    });

    // Password reset routes
    this.server.route({
      method: "POST",
      path: "/api/auth/request-reset",
      handler: AuthController.requestPasswordReset,
    });

    this.server.route({
      method: "POST",
      path: "/api/auth/reset-password",
      handler: AuthController.resetPassword,
    });

    // Protected route
    this.server.route({
      method: "POST",
      path: "/api/auth/logout",
      options: {
        pre: [{ method: verifyToken }],
      },
      handler: AuthController.logout,
    });
  };

  inferenceRoutes = () => {
    // Model inference route
    this.server.route({
      method: "POST",
      path: "/api/predict",
      options: {
        ...uploadMiddleware,
        pre: [{ method: verifyToken }, ...uploadMiddleware.pre],
      },
      handler: InferenceController.predict,
    });
  };
}

export default Application;
