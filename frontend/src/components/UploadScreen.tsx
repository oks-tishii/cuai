import React, { useRef } from 'react';
import { Upload, Image, Play, FileImage } from 'lucide-react';

interface UploadScreenProps {
  selectedImage: string | null;
  onImageSelect: (image: string) => void;
  onProcessImage: () => void;
  isProcessing: boolean;
}

export default function UploadScreen({ 
  selectedImage, 
  onImageSelect, 
  onProcessImage, 
  isProcessing 
}: UploadScreenProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageSelect(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageSelect(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">画像アップロード</h2>
          <p className="text-gray-400">異常検知を行いたい画像をアップロードしてください</p>
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
              className="border-2 border-dashed border-gray-600 rounded-xl p-12 text-center h-96 flex flex-col items-center justify-center relative overflow-hidden hover:border-gray-500 transition-colors"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              {selectedImage ? (
                <img
                  src={selectedImage}
                  alt="Selected"
                  className="max-w-full max-h-full object-contain rounded-lg"
                />
              ) : (
                <div className="space-y-6">
                  <FileImage className="w-20 h-20 text-gray-500 mx-auto" />
                  <div>
                    <p className="text-xl text-gray-300 mb-2">画像をドロップ</p>
                    <p className="text-gray-500">または上のボタンでファイルを選択</p>
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
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>

          {/* Preview and Controls */}
          <div className="space-y-6">
            <h3 className="text-xl font-semibold">プレビュー & 実行</h3>
            
            {selectedImage && (
              <div className="bg-gray-800 rounded-xl p-6">
                <div className="aspect-video bg-gray-700 rounded-lg mb-4 overflow-hidden">
                  <img
                    src={selectedImage}
                    alt="Preview"
                    className="w-full h-full object-contain"
                  />
                </div>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-700 rounded-lg p-3">
                      <div className="text-gray-400">ファイル形式</div>
                      <div className="font-medium">JPEG/PNG</div>
                    </div>
                    <div className="bg-gray-700 rounded-lg p-3">
                      <div className="text-gray-400">処理状態</div>
                      <div className="font-medium">
                        {isProcessing ? '処理中...' : '準備完了'}
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={onProcessImage}
                    disabled={isProcessing}
                    className="w-full py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600 rounded-lg transition-all flex items-center justify-center space-x-3 text-lg font-medium"
                  >
                    {isProcessing ? (
                      <>
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                        <span>PatchCore処理中...</span>
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
            )}

            {!selectedImage && (
              <div className="bg-gray-800 rounded-xl p-8 text-center">
                <Image className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400">画像を選択すると、プレビューと実行ボタンが表示されます</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}