/* 
*   Style Cam
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.Examples {

    using UnityEngine;
    using UnityEngine.UI;
    using NatSuite.Devices;
    using NatSuite.ML;
    using NatSuite.ML.Features;
    using NatSuite.ML.Vision;

    public class StyleCam : MonoBehaviour {

        [Header("UI")]
        public RawImage rawImage;
        public AspectRatioFitter aspectFitter;

        CameraDevice cameraDevice;
        Texture2D previewTexture;
        MLModelData modelData;
        MLModel model;
        FastNeuralStylePredictor predictor;
        RenderTexture styleTexture;

        async void Start () {
            // Request camera permissions
            if (!await MediaDeviceQuery.RequestPermissions<CameraDevice>()) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraDevice.previewResolution = (1280, 720);
            previewTexture = await cameraDevice.StartRunning();
            // Display the camera preview
            rawImage.texture = previewTexture;
            aspectFitter.aspectRatio = (float)previewTexture.width / previewTexture.height;
            // Get the style transfer model
            Debug.Log("Downloading model");
            modelData = await MLModelData.FromHub("@natsuite/style-mosaic");
            model = modelData.Deserialize();
            predictor = new FastNeuralStylePredictor(model);
            styleTexture = new RenderTexture(previewTexture.width, previewTexture.height, 0);
        }

        void Update () {
            // Check
            if (predictor == null)
                return;
            // Create input feature
            var input = new MLImageFeature(previewTexture);
            (input.mean, input.std) = modelData.normalization;
            // Stylize
            var styleImage = predictor.Predict(input);
            styleImage.Render(styleTexture);
            // Display
            rawImage.texture = styleTexture;
        }

        void OnDisable () {
            // Dispose the predictor and model
            //predictor?.Dispose();
            model?.Dispose();
            // Stop the camera preview
            if (cameraDevice?.running ?? false)
                cameraDevice.StopRunning();
        }
    }
}