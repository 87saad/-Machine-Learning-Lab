using NativeWebSocket;
using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

public class WebSocketObjectDetectionClient : MonoBehaviour
{
    [Header("Connection Settings")]
    [SerializeField] private string serverIP = "192.168.18.89";
    [SerializeField] private int serverPort = 8765;

    [Header("Platform")]
    [SerializeField] private bool isVRMode = false;

    [Header("Camera Settings")]
    [SerializeField] private int captureWidth = 640;
    [SerializeField] private int captureHeight = 480;
    [SerializeField] private int jpegQuality = 75;
    [SerializeField] private float sendInterval = 0.1f;

    // WebSocket
    private WebSocket websocket;
    private bool isConnected = false;

    // Camera
    private WebCamTexture webCamTexture; // Mobile
    private Camera vrCamera; // Quest Pro
    private OVRCameraRig ovrRig;

    // Processing
    private Texture2D captureTexture;
    private float lastSendTime;

    // Detection results
    public List<Detection> detections = new List<Detection>();

    [System.Serializable]
    public class Detection
    {
        public string className;
        public float confidence;
        public float x, y, width, height; // Normalized [0,1]
        public Vector3 worldPosition;
    }

    [System.Serializable]
    public class ServerResponse
    {
        public List<Detection> detections;
        public float inference_time;
        public string status;
    }

    async void Start()
    {
        try
        {

            // Auto-detect platform
#if UNITY_ANDROID && !UNITY_EDITOR
        isVRMode = UnityEngine.XR.XRSettings.isDeviceActive;
#endif

            Debug.Log($"Starting in {(isVRMode ? "VR" : "Mobile")} mode");

            // Setup camera
            if (isVRMode)
            {
                SetupVRCamera();
            }
            else
            {
                SetupMobileCamera();
            }

            captureTexture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);

            // Connect to server
            await ConnectToServer();
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in Start: {e.Message}");
            isConnected = false;
        }
    }

    void SetupMobileCamera()
    {
#if UNITY_ANDROID || UNITY_IOS
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            Application.RequestUserAuthorization(UserAuthorization.WebCam);
        }
#endif

        if (WebCamTexture.devices.Length > 0)
        {
            string deviceName = "";
            foreach (var device in WebCamTexture.devices)
            {
                if (!device.isFrontFacing)
                {
                    deviceName = device.name;
                    break;
                }
            }

            if (string.IsNullOrEmpty(deviceName))
                deviceName = WebCamTexture.devices[0].name;

            webCamTexture = new WebCamTexture(deviceName, 1280, 720, 30);
            webCamTexture.Play();

            Debug.Log($"Mobile camera started: {deviceName}");
        }
    }

    void SetupVRCamera()
    {
        ovrRig = FindAnyObjectByType<OVRCameraRig>();

        if (ovrRig != null)
        {
            vrCamera = ovrRig.centerEyeAnchor.GetComponent<Camera>();

            if (OVRManager.instance != null)
            {
                OVRManager.instance.isInsightPassthroughEnabled = true;
            }

            Debug.Log("VR camera initialized");
        }
        else
        {
            Debug.LogError("OVRCameraRig not found!");
        }
    }

    async System.Threading.Tasks.Task ConnectToServer()
    {
        string url = $"ws://{serverIP}:{serverPort}";
        Debug.Log($"Connecting to {url}...");

        websocket = new WebSocket(url);

        websocket.OnOpen += () =>
        {
            Debug.Log("WebSocket Connected!");
            isConnected = true;
        };

        websocket.OnError += (e) =>
        {
            Debug.LogError($"WebSocket Error: {e}");
            isConnected = false;
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log($"WebSocket Closed: {e}");
            isConnected = false;
        };

        websocket.OnMessage += (bytes) =>
        {
            string json = Encoding.UTF8.GetString(bytes);
            HandleServerResponse(json);
        };

        await websocket.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
#endif

        // Send frames at interval
        if (isConnected && Time.time - lastSendTime >= sendInterval)
        {
            SendFrameToServer();
            lastSendTime = Time.time;
        }
    }

    void SendFrameToServer()
    {
        // Capture frame
        Texture2D frame = CaptureFrame();

        if (frame == null)
            return;

        // Encode to JPEG
        byte[] jpegData = frame.EncodeToJPG(jpegQuality);

        // Send via WebSocket
        if (websocket.State == WebSocketState.Open)
        {
            websocket.Send(jpegData);
        }
    }

    Texture2D CaptureFrame()
    {
        if (isVRMode)
        {
            return CaptureVRFrame();
        }
        else
        {
            return CaptureMobileFrame();
        }
    }

    Texture2D CaptureMobileFrame()
    {
        if (webCamTexture == null || !webCamTexture.isPlaying)
            return null;

        RenderTexture rt = RenderTexture.GetTemporary(captureWidth, captureHeight);
        Graphics.Blit(webCamTexture, rt);

        RenderTexture.active = rt;
        captureTexture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        captureTexture.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        return captureTexture;
    }

    Texture2D CaptureVRFrame()
    {
        if (vrCamera == null)
            return null;

        RenderTexture rt = RenderTexture.GetTemporary(captureWidth, captureHeight, 24);
        RenderTexture currentRT = RenderTexture.active;

        vrCamera.targetTexture = rt;
        vrCamera.Render();

        RenderTexture.active = rt;
        captureTexture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        captureTexture.Apply();

        vrCamera.targetTexture = null;
        RenderTexture.active = currentRT;
        RenderTexture.ReleaseTemporary(rt);

        return captureTexture;
    }

    void HandleServerResponse(string json)
    {
        try
        {
            ServerResponse response = JsonUtility.FromJson<ServerResponse>(json);

            if (response.status == "success")
            {
                detections = response.detections;

                // Calculate world positions for VR
                if (isVRMode)
                {
                    CalculateWorldPositions();
                }

                Debug.Log($"Received {detections.Count} detections (inference: {response.inference_time:F3}s)");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to parse response: {e.Message}");
        }
    }

    void CalculateWorldPositions()
    {
        if (vrCamera == null)
            return;

        foreach (Detection det in detections)
        {
            // Convert normalized bbox to screen space
            float screenX = (det.x + det.width / 2) * Screen.width;
            float screenY = (1 - (det.y + det.height / 2)) * Screen.height;

            Ray ray = vrCamera.ScreenPointToRay(new Vector3(screenX, screenY, 0));
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, 10f))
            {
                det.worldPosition = hit.point;
            }
            else
            {
                det.worldPosition = ray.GetPoint(2f);
            }
        }
    }

    void OnGUI()
    {
        // Display status
        GUI.Label(new Rect(10, 10, 400, 30),
            $"Mode: {(isVRMode ? "VR" : "Mobile")} | Status: {(isConnected ? "Connected" : "Disconnected")}");
        GUI.Label(new Rect(10, 40, 400, 30),
            $"Server: {serverIP}:{serverPort} | FPS: {(1f / Time.deltaTime):F0}");
        GUI.Label(new Rect(10, 70, 400, 30),
            $"Detections: {detections.Count}");

        // Show camera feed on mobile
        if (!isVRMode && webCamTexture != null && webCamTexture.isPlaying)
        {
            GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), webCamTexture);
        }

        // Draw detections
        DrawDetections();
    }

    void DrawDetections()
    {
        GUI.color = Color.green;

        foreach (Detection det in detections)
        {
            // Convert normalized coords to screen space
            Rect screenRect = new Rect(
                det.x * Screen.width,
                det.y * Screen.height,
                det.width * Screen.width,
                det.height * Screen.height
            );

            // Draw bounding box
            GUI.Box(screenRect, "");

            // Draw label
            string label = $"{det.className} {det.confidence:P0}";
            GUI.Label(new Rect(screenRect.x, screenRect.y - 20, 200, 20), label);
        }

        GUI.color = Color.white;
    }

    async void OnDestroy()
    {
        if (websocket != null && websocket.State == WebSocketState.Open)
        {
            await websocket.Close();
        }

        if (webCamTexture != null)
        {
            webCamTexture.Stop();
            Destroy(webCamTexture);
        }

        if (captureTexture != null)
        {
            Destroy(captureTexture);
        }
    }

    async void OnApplicationQuit()
    {
        if (websocket != null && websocket.State == WebSocketState.Open)
        {
            await websocket.Close();
        }
    }
}