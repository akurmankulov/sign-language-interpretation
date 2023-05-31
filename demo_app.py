from streamlit_webrtc import WebRtcMode, webrtc_streamer
from turn import get_ice_servers

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    # video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# class MyVideoTransformer(VideoProcessorBase):
#     def __init__(self):
#         self.hand_detector = HandDetector(maxHands=1)

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         hands, image_hand = self.hand_detector.findHands(frame)
#         return hands, image_hand

        # img = frame.to_image()
        # imgOutput = img.copy()
        # hands, img = detector.findHands(img)
        # if hands:
        #     hand = hands[0]
        #     x, y, w, h = hand['bbox']

        #     imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        #     imgCrop = img[y-offset : y+h+offset, x-offset : x+w+offset]

        #     #imgCropShape = imgCrop.shape

        #     aspectRatio = h/w
        # # try:
        #     if aspectRatio > 1:
        #         k = imgSize/h
        #         wCal = math.ceil(k*w)
        #         imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        #         #imgResizeShape = imgResize.shape
        #         wGap = math.ceil((imgSize-wCal)/2)
        #         imgWhite[:, wGap:wCal+wGap] = imgResize
        #         index = int(np.argmax(model.predict(imgWhite.reshape(-1,300,300,3)), axis=1))

        #     else:
        #         k = imgSize/w
        #         hCal = math.ceil(k*h)
        #         imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        #         #imgResizeShape = imgResize.shape
        #         hGap = math.ceil((imgSize-hCal)/2)
        #         imgWhite[hGap:hCal+hGap, :] = imgResize
        #         index = int(np.argmax(model.predict(imgWhite.reshape(-1,300,300,3)), axis=1))
        # #except:
        #     #   continue

        #     cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset), (255, 0, 255), cv2.FILLED)
        #     cv2.putText(imgOutput, labels[index], (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
        #     cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        #     #cv2.imshow("Image", imgOutput)
        #     frame_tr = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        #    return av.VideoFrame.from_image(frame_tr)
