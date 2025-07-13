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
  timestamp: string; // Date object is not directly serializable
}

function App() {
  const [currentScreen, setCurrentScreen] = useState("upload");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResult, setDetectionResult] =
    useState<DetectionResult | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [history, setHistory] = useState<DetectionResult[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  const handleImageSelect = (imageFile: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64Image = e.target?.result as string;
      setSelectedImage(base64Image);
    };
    reader.readAsDataURL(imageFile);
  };

  const handleProcessImage = async () => {
    if (!selectedImage) return;
    setIsProcessing(true);
    try {
      const result = await window.pywebview.api.process_image(selectedImage);
      console.log("Detection result from Python:", result);
      setDetectionResult(result);
      setHistory((prev) => [result, ...prev.slice(0, 19)]);
      setCurrentScreen("analysis");
    } catch (error) {
      console.error("Error processing image:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRetrainModel = async () => {
    setIsTraining(true);
    try {
      const result = await window.pywebview.api.retrain_model();
      console.log("Retrain result:", result);
      // Optionally, show a notification to the user
    } catch (error) {
      console.error("Error retraining model:", error);
    } finally {
      setIsTraining(false);
    }
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
            onRetrainModel={handleRetrainModel}
            isTraining={isTraining}
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