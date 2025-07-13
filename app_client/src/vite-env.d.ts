/// <reference types="vite/client" />

declare global {
  interface Window {
    pywebview: {
      api: {
        process_image: (imageBase64: string) => Promise<DetectionResult>;
        retrain_model: () => Promise<{ status: string; message: string }>;
      };
    };
  }
}

