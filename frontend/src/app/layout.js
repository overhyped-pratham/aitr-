import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "DriveSafe AI — AI-Powered Driver Safety & Fatigue Prevention",
  description: "Real-time webcam driver drowsiness monitoring, custom CNN fatigue score aggregation, Gemini-powered safety coaching, and geo-fenced emergency assistance.",
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossOrigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossOrigin="anonymous"></script>
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
