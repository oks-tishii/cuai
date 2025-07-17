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
  processingTime: number;
  confidence: number;
  numAnomalyRegions: number;
  maxAnomalyScore: number;
}

function App() {
  const [currentScreen, setCurrentScreen] = useState("upload");
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>(
    []
  );
  const [threshold, setThreshold] = useState(0.5);
  const [history, setHistory] = useState<DetectionResult[]>([]);

  const handleImagesSelect = (images: File[]) => {
    setSelectedImages(images);
  };

  const handleProcessImages = async () => {
    if (selectedImages.length === 0) return;

    setIsProcessing(true);

    const formData = new FormData();
    selectedImages.forEach((file) => {
      formData.append("files", file, file.name);
    });
    formData.append("threshold", String(threshold));

    try {
      const response = await fetch("/detect-anomaly-batch", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Anomaly detection failed: ${response.statusText}`);
      }

      const resultData = await response.json();

      const newResults = await Promise.all(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        resultData.results.map(async (res: any, index: number) => {
          const originalImageFile = selectedImages[index];
          const imageBase64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(originalImageFile);
          });

          return {
            id: res.image_path || `${Date.now()}-${index}`,
            image: imageBase64 as string,
            anomalyScore: res.anomaly_score,
            isAnomalous: res.anomaly_score > threshold, // Use the score from the response
            heatmap: `data:image/png;base64,${res.heatmap_base64}`,
            markedImage: `data:image/png;base64,${res.marking_base64}`,
            timestamp: new Date(),
            processingTime: res.processing_time,
            confidence: res.confidence,
            numAnomalyRegions: res.num_anomaly_regions,
            maxAnomalyScore: res.max_anomaly_score,
          };
        })
      );

      setDetectionResults(newResults);
      setHistory((prev) => [...newResults, ...prev].slice(0, 20));
      setCurrentScreen("analysis");
    } catch (error) {
      console.error("Error during anomaly detection:", error);
      // Consider showing an error message to the user
    } finally {
      setIsProcessing(false);
      setSelectedImages([]);
    }
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
