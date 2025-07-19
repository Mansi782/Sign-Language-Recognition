// static/js/camera.js
document.addEventListener('DOMContentLoaded', function() {
    const openCameraBtn = document.getElementById("openCameraBtn");
    const captureGestureBtn = document.getElementById("captureGestureBtn");
    const cameraFeed = document.getElementById("cameraFeed");
    const predictedGesture = document.getElementById("predictedGesture");

    async function openCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraFeed.srcObject = stream;
        } catch (error) {
            console.error("Error accessing the camera:", error);
        }
    }

    async function captureAndPredictGesture() {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append("file", blob, "capture.png");

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                if (result.gesture) {
                    predictedGesture.textContent = result.gesture;
                } else {
                    predictedGesture.textContent = "Error";
                }
            } catch (error) {
                console.error("Error predicting gesture:", error);
                predictedGesture.textContent = "Error";
            }
        });
    }

    openCameraBtn.addEventListener("click", openCamera);
    captureGestureBtn.addEventListener("click", captureAndPredictGesture);
});