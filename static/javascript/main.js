let base64Image;
$("#image-selector").change(function() {
    let reader = new FileReader();
    reader.onload = function(e) {
        let dataURL = reader.result;
        $('#selected-image').attr("src", dataURL);
        base64Image = dataURl.replace("data:image/png;base64,","");
        console.log(base64Image);
    }
    reader.readAsDataURL($("#image-selector")[0].files[0]);
    $("#dog-prediction").text("");
    $("#cat-prediction").text("");
});

$("#predict-button").click(function(event){
    let message = {
        image: base64Image
    }
    console.log(message)
    $.get("/predict", JSON.stringify(message), function(response){
        $("#dog-predictions").text(response.prediction.dog.toFixed(6));
        $("#cat-prediction").text(response.prediction.cat.toFixed(6));
        console.log(response)
    });
});