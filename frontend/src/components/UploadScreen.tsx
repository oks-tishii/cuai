import React, { useRef } from "react";
import { Upload, Play, FileImage, X } from "lucide-react";

interface UploadScreenProps {
  selectedImages: File[];
  onImagesSelect: (images: File[]) => void;
  onProcessImages: () => void;
  isProcessing: boolean;
}

export default function UploadScreen({
  selectedImages,
  onImagesSelect,
  onProcessImages,
  isProcessing,
}: UploadScreenProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      const newImages = Array.from(files);
      onImagesSelect([...selectedImages, ...newImages]);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files) {
      const imageFiles = Array.from(files).filter((file) =>
        file.type.startsWith("image/")
      );
      onImagesSelect([...selectedImages, ...imageFiles]);
    }
  };

  const handleRemoveImage = (index: number) => {
    const newImages = [...selectedImages];
    newImages.splice(index, 1);
    onImagesSelect(newImages);
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">画像アップロード</h2>
          <p className="text-gray-400">
            異常検知を行いたい画像をまとめてアップロードしてください
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Area */}
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">画像選択</h3>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center space-x-2"
              >
                <Upload className="w-4 h-4" />
                <span>ファイル選択</span>
              </button>
            </div>

            <div
              className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center h-96 flex flex-col items-center justify-center relative overflow-hidden hover:border-gray-500 transition-colors"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              {selectedImages.length > 0 ? (
                <div className="grid grid-cols-3 gap-4 overflow-y-auto h-full w-full">
                  {selectedImages.map((image, index) => (
                    <div key={index} className="relative group">
                      <img
                        src={URL.createObjectURL(image)}
                        alt={`Selected ${index}`}
                        className="w-full h-full object-cover rounded-lg"
                      />
                      <button
                        onClick={() => handleRemoveImage(index)}
                        className="absolute top-1 right-1 bg-red-600 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-6">
                  <FileImage className="w-20 h-20 text-gray-500 mx-auto" />
                  <div>
                    <p className="text-xl text-gray-300 mb-2">
                      画像をドラッグ＆ドロップ
                    </p>
                    <p className="text-gray-500">
                      または上のボタンでファイルを選択
                    </p>
                  </div>
                  <div className="text-sm text-gray-600">
                    対応形式: JPG, PNG, GIF (最大10MB)
                  </div>
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>

          {/* Controls */}
          <div className="space-y-6">
            <h3 className="text-xl font-semibold">実行コントロール</h3>

            <div className="bg-gray-800 rounded-xl p-6">
              <div className="grid grid-cols-2 gap-4 text-sm mb-6">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="text-gray-400">選択枚数</div>
                  <div className="font-medium">{selectedImages.length} 枚</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="text-gray-400">処理状態</div>
                  <div className="font-medium">
                    {isProcessing
                      ? "処理中..."
                      : selectedImages.length > 0
                      ? "準備完了"
                      : "画像未選択"}
                  </div>
                </div>
              </div>

              <button
                onClick={onProcessImages}
                disabled={isProcessing || selectedImages.length === 0}
                className="w-full py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600 rounded-lg transition-all flex items-center justify-center space-x-3 text-lg font-medium"
              >
                {isProcessing ? (
                  <>
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                    <span>処理中...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-6 h-6" />
                    <span>異常検知を実行</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
