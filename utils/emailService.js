import nodemailer from "nodemailer";
import { configDotenv } from "dotenv";

// Load environment variables
configDotenv();

class EmailService {
  constructor() {
    // Gmail SMTP configuration
    this.transporter = nodemailer.createTransport({
      service: "gmail", // Using Gmail service
      host: "smtp.gmail.com",
      port: 465,
      secure: true, // Use SSL
      auth: {
        user: process.env.GMAIL_USER,
        pass: process.env.GMAIL_APP_PASSWORD, // Use App Password, not regular password
      },
      tls: {
        rejectUnauthorized: false, // For development environments
      },
    });

    // Verify connection configuration
    this.verifyConnection();
  }

  async sendPasswordResetEmail(email, resetToken, userName) {
    try {
      if (!process.env.GMAIL_USER || !process.env.GMAIL_APP_PASSWORD) {
        throw new Error(
          "Email configuration is missing. Please check your environment variables."
        );
      }

      const resetLink = `${process.env.FRONTEND_URL}/reset-password?token=${resetToken}`;

      const mailOptions = {
        from: `"AgriScan" <${process.env.GMAIL_USER}>`, // Using Gmail address
        to: email,
        subject: "Reset Password - AgriScan",
        html: `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f7fafc;">
            <div style="background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
              <h2 style="color: #2C5282; margin-bottom: 20px;">Reset Your Password</h2>
              <p style="color: #4a5568;">Hello ${userName},</p>
              <p style="color: #4a5568;">We received a request to reset your password. Click the button below to create a new password:</p>
              <div style="margin: 30px 0; text-align: center;">
                <a href="${resetLink}" 
                   style="background-color: #4299E1; 
                          color: white; 
                          padding: 12px 24px; 
                          text-decoration: none; 
                          border-radius: 4px;
                          display: inline-block;
                          font-weight: bold;">
                  Reset Password
                </a>
              </div>
              <p style="color: #4a5568; font-weight: bold;">This link will expire in 30 minutes for security reasons.</p>
              <p style="color: #e53e3e; margin-top: 20px;">If you didn't request this password reset, please ignore this email or contact support if you have concerns.</p>
              <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
                <p style="color: #718096; font-size: 14px;">
                  Best regards,<br>
                  The AgriScan Team
                </p>
              </div>
            </div>
            <div style="text-align: center; margin-top: 20px; color: #718096; font-size: 12px;">
              <p>This is an automated message, please do not reply to this email.</p>
            </div>
          </div>
        `,
      };

      const info = await this.transporter.sendMail(mailOptions);
      console.log("✅ Password reset email sent successfully:", info.messageId);
      return true;
    } catch (error) {
      console.error("❌ Error sending password reset email:", error.message);
      throw new Error(`Failed to send password reset email: ${error.message}`);
    }
  }

  async verifyConnection() {
    try {
      await this.transporter.verify();
      console.log("✅ Email service connected successfully");
      return true;
    } catch (error) {
      console.error("❌ Email service connection error:", error.message);
      console.log("Please check your email configuration in .env file");
      return false;
    }
  }
}

export default new EmailService();
