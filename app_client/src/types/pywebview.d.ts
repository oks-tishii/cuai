export {};

declare global {
  interface Window {
    pywebview: {
      api: {
        process_image: (image: string) => Promise<DetectionResult>;
      };
    };
  }
}
