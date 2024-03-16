const ctx = new AudioContext();
let audio;

document.querySelector("#denoiseForm").addEventListener("submit", function (e) {
  e.preventDefault();

  var formData = new FormData();
  var audioFile = document.querySelector('input[type="file"]').files[0];

  if (!audioFile) {
    console.error("No file selected");
    return;
  }

  formData.append("audio", audioFile);

  fetch("/denoise", {
    method: "POST",
    body: formData,
  })
    .then((data) => {
      console.log(data);
      return data.arrayBuffer();
    })
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
    .then((decodedAudio) => {
      audio = decodedAudio;
      console.log(audio);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
});


function playbackAudio() {
  const playSound = ctx.createBufferSource();
  playSound.buffer = audio;
  playSound.connect(ctx.destination);
  playSound.start(ctx.currentTime);
}

document
  .querySelector("#inputGroupFile01")
  .addEventListener("change", function (e) {
    var audioFile = e.target.files[0];
    if (audioFile) {
      e.target.labels[0].innerHTML = audioFile.name;
    }
    console.log(e);

    var audioElement = document.getElementById("audioTag");
    audioElement.src = URL.createObjectURL(audioFile);
  });

document.querySelector("#playAudio").addEventListener("click", () => {
  var audioElement = document.getElementById("audioTag");
  audioElement.play();
  console.log(audioElement);
});


document.querySelector("#outputAudio").addEventListener("click", (e) => {
  playbackAudio()
  console.log("play");
})
