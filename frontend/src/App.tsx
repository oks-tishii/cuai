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
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResult, setDetectionResult] =
    useState<DetectionResult | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [history, setHistory] = useState<DetectionResult[]>([]);

  const handleImageSelect = (image: string) => {
    setSelectedImage(image);
  };

  const handleProcessImage = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);

    // Simulate PatchCore processing
    setTimeout(() => {
      const score = Math.random();
      const result: DetectionResult = {
        id: Date.now().toString(),
        image: selectedImage,
        anomalyScore: score,
        isAnomalous: score > threshold,
        heatmap: `https://images.pexels.com/photos/8566526/pexels-photo-8566526.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop`,
        markedImage: `https://images.pexels.com/photos/8566527/pexels-photo-8566527.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop`,
        timestamp: new Date(),
      };

      setDetectionResult(result);
      setHistory((prev) => [result, ...prev.slice(0, 19)]);
      setIsProcessing(false);

      // Auto-switch to analysis screen after processing
      setCurrentScreen("analysis");
    }, 3000);
  };

  const handleSelectResult = (result: DetectionResult) => {
    setDetectionResult(result);
    setCurrentScreen("analysis");
  };

  const handleDeleteResult = (id: string) => {
    setHistory((prev) => prev.filter((r) => r.id !== id));
    if (detectionResult?.id === id) {
      setDetectionResult(null);
    }
  };

  const renderCurrentScreen = () => {
    switch (currentScreen) {
      case "upload":
        return (
          <UploadScreen
            selectedImage={selectedImage}
            onImageSelect={handleImageSelect}
            onProcessImage={handleProcessImage}
            isProcessing={isProcessing}
          />
        );
      case "analysis":
        return (
          <AnalysisScreen
            detectionResult={detectionResult}
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
