import Auth from "../models/Auth.js";
// import { verifyToken } from "../middleware/auth";
import bcryptjs from "bcryptjs";
import jwt from "jsonwebtoken";
import crypto from "crypto";

class AuthController {
  register = async (req, h) => {
    try {
      const { email, userName, password } = req.payload;

      if (!email || !userName || !password) {
        return h
          .response({
            status: 400,
            message: "Email, username and password are required",
          })
          .code(400);
      }

      const isAlreadyRegistered = await Auth.findOne({ email });
      if (isAlreadyRegistered) {
        return h
          .response({
            status: 400,
            message: "This email is already registered",
          })
          .code(400);
      }

      const hash = await bcryptjs.hash(password, 10);
      const newAuth = new Auth({
        email,
        userName,
        password: hash,
      });

      await newAuth.save();

      return h
        .response({
          status: 201,
          data: {
            email: newAuth.email,
            userName: newAuth.userName,
            createdAt: newAuth.createdAt,
          },
          message: "Account successfully registered",
        })
        .code(201);
    } catch (error) {
      console.error("Register error:", error);
      return h
        .response({
          status: 500,
          message: "Internal server error",
        })
        .code(500);
    }
  };

  login = async (req, h) => {
    try {
      const { userName, password } = req.payload;

      if (!userName || !password) {
        return h
          .response({
            status: 400,
            message: "Username and password are required",
          })
          .code(400);
      }

      const user = await Auth.findOne({ userName });
      if (!user) {
        return h
          .response({
            status: 404,
            message: "Account not found",
          })
          .code(404);
      }

      const isValidPassword = await bcryptjs.compare(password, user.password);
      if (!isValidPassword) {
        return h
          .response({
            status: 400,
            message: "Invalid credentials",
          })
          .code(400);
      }

      if (!process.env.JWT_SECRET) {
        console.error("JWT_SECRET is not defined in environment variables");
        return h
          .response({
            status: 500,
            message: "Internal server error",
          })
          .code(500);
      }

      const token = jwt.sign(
        {
          _id: user._id,
          email: user.email,
          userName: user.userName,
        },
        process.env.JWT_SECRET,
        { expiresIn: "1d" }
      );

      user.token = token;
      await user.save();

      return h
        .response({
          status: 200,
          data: {
            email: user.email,
            userName: user.userName,
            token: user.token,
          },
          message: "Login successful",
        })
        .code(200);
    } catch (error) {
      console.error("Login error:", error);
      return h
        .response({
          status: 500,
          message: "Internal server error",
        })
        .code(500);
    }
  };

  logout = async (req, h) => {
    try {
      const { _id } = req.user;
      const user = await Auth.findById(_id);

      if (!user) {
        return h
          .response({
            status: 404,
            message: "Account not found",
          })
          .code(404);
      }

      user.token = null;
      await user.save();

      return h
        .response({
          status: 200,
          message: "Logout successful",
        })
        .code(200);
    } catch (error) {
      console.error("Logout error:", error);
      return h
        .response({
          status: 500,
          message: "Internal server error",
        })
        .code(500);
    }
  };

  requestPasswordReset = async (req, h) => {
    try {
      const { email } = req.payload;

      if (!email) {
        return h
          .response({
            status: 400,
            message: "Email is required",
          })
          .code(400);
      }

      const user = await Auth.findOne({ email });
      if (!user) {
        return h
          .response({
            status: 404,
            message: "User not found",
          })
          .code(404);
      }

      // Generate reset token
      const resetToken = crypto.randomBytes(32).toString("hex");
      const hashedToken = crypto
        .createHash("sha256")
        .update(resetToken)
        .digest("hex");

      // Save reset token and expiry
      user.resetPasswordToken = hashedToken;
      user.resetPasswordExpires = new Date(Date.now() + 30 * 60 * 1000); // 30 minutes
      await user.save();

      // In a real application, you would send this token via email
      // For development, we'll return it in the response
      return h
        .response({
          status: 200,
          message: "Password reset token generated successfully",
          data: {
            resetToken,
            email: user.email,
          },
        })
        .code(200);
    } catch (error) {
      console.error("Password reset request error:", error);
      return h
        .response({
          status: 500,
          message: "Internal server error",
        })
        .code(500);
    }
  };

  resetPassword = async (req, h) => {
    try {
      const { token, newPassword } = req.payload;

      if (!token || !newPassword) {
        return h
          .response({
            status: 400,
            message: "Token and new password are required",
          })
          .code(400);
      }

      // Hash the token for comparison
      const hashedToken = crypto
        .createHash("sha256")
        .update(token)
        .digest("hex");

      const user = await Auth.findOne({
        resetPasswordToken: hashedToken,
        resetPasswordExpires: { $gt: Date.now() },
      });

      if (!user) {
        return h
          .response({
            status: 400,
            message: "Invalid or expired reset token",
          })
          .code(400);
      }

      // Validate new password
      const passwordRegex = /^(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;
      if (!passwordRegex.test(newPassword)) {
        return h
          .response({
            status: 400,
            message:
              "Password must be at least 8 characters, include an uppercase letter, a number, and a special character",
          })
          .code(400);
      }

      // Hash new password and update user
      const hash = await bcryptjs.hash(newPassword, 10);
      user.password = hash;
      user.resetPasswordToken = null;
      user.resetPasswordExpires = null;
      await user.save();

      return h
        .response({
          status: 200,
          message: "Password reset successful",
        })
        .code(200);
    } catch (error) {
      console.error("Password reset error:", error);
      return h
        .response({
          status: 500,
          message: "Internal server error",
        })
        .code(500);
    }
  };
}

export default new AuthController();
