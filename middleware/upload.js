import multer from "multer";
import path from "path";
import fs from "fs";

// Configure storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Create uploads directory if it doesn't exist
    const uploadDir = path.join(process.cwd(), "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname);
    cb(null, `image-${uniqueSuffix}${ext}`);
  },
});

// File filter
const fileFilter = (req, file, cb) => {
  // Accept images only
  const allowedTypes = /jpeg|jpg|png|gif/;
  const extname = allowedTypes.test(
    path.extname(file.originalname).toLowerCase()
  );
  const mimetype = allowedTypes.test(file.mimetype);

  if (extname && mimetype) {
    cb(null, true);
  } else {
    cb(new Error(`Only image files (${allowedTypes.source}) are allowed!`));
  }
};

// Create multer instance
const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB max file size
  },
});

// Middleware for Hapi
export const uploadMiddleware = {
  payload: {
    output: "file",
    parse: true,
    multipart: true,
    maxBytes: 5 * 1024 * 1024, // 5MB max file size
    allow: "multipart/form-data",
    timeout: false, // Disable timeout for file uploads
  },
  pre: [
    {
      method: async (request, h) => {
        try {
          console.log("📤 Processing file upload");

          // Check if payload exists
          if (!request.payload) {
            console.error("❌ No payload received in upload middleware");
            return h
              .response({
                status: 400,
                message: "No payload received",
              })
              .code(400)
              .takeover();
          }

          const file = request.payload.image;

          // Check if file exists in payload
          if (!file) {
            console.error("❌ No image file in payload");
            return h
              .response({
                status: 400,
                message: "No image file provided",
              })
              .code(400)
              .takeover();
          }

          // Log file details
          console.log("📦 Received file:", {
            filename: file.filename,
            mimetype: file.headers["content-type"],
            size: file.bytes,
          });

          // Validate file type
          const allowedTypes = /jpeg|jpg|png|gif/;
          const extname = allowedTypes.test(
            path.extname(file.filename).toLowerCase()
          );
          const mimetype = allowedTypes.test(file.headers["content-type"]);

          if (!extname || !mimetype) {
            console.error(
              "❌ Invalid file type:",
              file.headers["content-type"]
            );
            return h
              .response({
                status: 400,
                message: `Only image files (${allowedTypes.source}) are allowed`,
              })
              .code(400)
              .takeover();
          }

          // Create uploads directory if it doesn't exist
          const uploadDir = path.join(process.cwd(), "uploads");
          if (!fs.existsSync(uploadDir)) {
            console.log("📁 Creating uploads directory");
            fs.mkdirSync(uploadDir, { recursive: true });
          }

          // Generate unique filename
          const uniqueSuffix =
            Date.now() + "-" + Math.round(Math.random() * 1e9);
          const ext = path.extname(file.filename);
          const newFilename = `image-${uniqueSuffix}${ext}`;
          const filePath = path.join(uploadDir, newFilename);

          // Move file to uploads directory
          try {
            console.log("📋 Moving file to:", filePath);
            // Use copyFile and unlink instead of rename for cross-device compatibility
            await fs.promises.copyFile(file.path, filePath);
            await fs.promises.unlink(file.path);

            // Update file information
            file.path = filePath;
            file.filename = newFilename;

            console.log("✅ File upload processed successfully");
            return h.continue;
          } catch (moveError) {
            console.error("❌ Error moving file:", moveError);
            return h
              .response({
                status: 500,
                message: "Failed to process uploaded file",
              })
              .code(500)
              .takeover();
          }
        } catch (error) {
          console.error("❌ Upload middleware error:", error);
          return h
            .response({
              status: 500,
              message: `File upload failed: ${error.message}`,
            })
            .code(500)
            .takeover();
        }
      },
    },
  ],
};
