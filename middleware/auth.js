import jwt from "jsonwebtoken";

export const verifyToken = async (request, h) => {
  try {
    const token = request.headers.authorization?.replace("Bearer ", "");

    if (!token) {
      return h
        .response({
          status: 401,
          message: "No token provided",
        })
        .code(401)
        .takeover();
    }

    if (!process.env.JWT_SECRET) {
      console.error("JWT_SECRET is not defined in environment variables");
      return h
        .response({
          status: 500,
          message: "Internal server error",
        })
        .code(500)
        .takeover();
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    request.user = decoded;

    return h.continue;
  } catch (error) {
    return h
      .response({
        status: 401,
        message: "Invalid token",
      })
      .code(401)
      .takeover();
  }
};
