import { useState } from "react";
import Navigation from "./components/Navigation";
import UploadScreen from "./components/UploadScreen";
import AnalysisScreen from "./components/AnalysisScreen";
import HistoryScreen from "./components/HistoryScreen";
import SettingsScreen from "./components/SettingsScreen";

interface DetectionResult {
  id: string;
  image: string;
  anomalyScore: number;
  isAnomalous: boolean;
  heatmap: string;
  markedImage: string;
  timestamp: Date;
}

function App() {
  const [currentScreen, setCurrentScreen] = useState("upload");
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResults, setDetectionResults] = useState<
    DetectionResult[]
  >([]);
  const [threshold, setThreshold] = useState(0.5);
  const [history, setHistory] = useState<DetectionResult[]>([]);

  const handleImagesSelect = (images: string[]) => {
    setSelectedImages(images);
  };

  const handleProcessImages = async () => {
    if (selectedImages.length === 0) return;

    setIsProcessing(true);
    const results: DetectionResult[] = [];

    // Simulate PatchCore processing for each image
    for (const image of selectedImages) {
      await new Promise((resolve) => setTimeout(resolve, 1000)); // Simulate network delay
      const score = Math.random();
      const result: DetectionResult = {
        id: Date.now().toString() + image,
        image: image,
        anomalyScore: score,
        isAnomalous: score > threshold,
        heatmap: `https://images.pexels.com/photos/8566526/pexels-photo-8566526.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop`,
        markedImage: `https://images.pexels.com/photos/8566527/pexels-photo-8566527.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop`,
        timestamp: new Date(),
      };
      results.push(result);
    }

    setDetectionResults(results);
    setHistory((prev) => [...results, ...prev].slice(0, 20));
    setIsProcessing(false);
    setSelectedImages([]);

    // Auto-switch to analysis screen after processing
    setCurrentScreen("analysis");
  };

  const handleSelectResult = (result: DetectionResult) => {
    setDetectionResults([result]);
    setCurrentScreen("analysis");
  };

  const handleDeleteResult = (id: string) => {
    setHistory((prev) => prev.filter((r) => r.id !== id));
    setDetectionResults((prev) => prev.filter((r) => r.id !== id));
  };

  const renderCurrentScreen = () => {
    switch (currentScreen) {
      case "upload":
        return (
          <UploadScreen
            selectedImages={selectedImages}
            onImagesSelect={handleImagesSelect}
            onProcessImages={handleProcessImages}
            isProcessing={isProcessing}
          />
        );
      case "analysis":
        return (
          <AnalysisScreen
            detectionResults={detectionResults}
            threshold={threshold}
          />
        );
      case "history":
        return (
          <HistoryScreen
            history={history}
            onSelectResult={handleSelectResult}
            onDeleteResult={handleDeleteResult}
          />
        );
      case "settings":
        return (
          <SettingsScreen
            threshold={threshold}
            onThresholdChange={setThreshold}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <Navigation
        currentScreen={currentScreen}
        onScreenChange={setCurrentScreen}
      />
      {renderCurrentScreen()}
    </div>
  );
}

export default App;
