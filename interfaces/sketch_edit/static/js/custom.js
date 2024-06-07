var canvas = document.getElementById('myCanvas');
var context = canvas.getContext('2d');

var images = {};
var imagePositions = {}
var selectedImage = null;
var selectedLayer = null; // for copy and delete
var dragOffsetX, dragOffsetY;
var loadedImageName = ""// if we upload sketch from UI, we will store the name here
var hasInpainted = false
var priorToInpaintingLayerData = null

/** MAIN FUNCTION */

/** UI FUNCTIONS **/
function drawImages() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    // console.log('In drawImages: Drawing images:', images, imagePositions)
    for (var key in images) {
        if (images.hasOwnProperty(key)) {
            var pos = imagePositions[key];
            if (pos) {
                // console.log('Drawing image:', key, pos.x, pos.y, pos.width, pos.height)
                context.drawImage(images[key], pos.x, pos.y, pos.width, pos.height);
            }
        }
    }
}

function loadImages(sources, callback) {
    // var images = {};
    var loadedImages = 0;
    var numImages = 0;
    // get num of sources
    for (var src in sources) {
        numImages++;
    }
    for (var src in sources) {
        images[src] = new Image();
        images[src].onload = function () {
            if (++loadedImages >= numImages) {
                callback(images);
            }
        };
        images[src].src = sources[src];
    }
}


canvas.onmousedown = function (event) {
    var mouseX = event.clientX - canvas.offsetLeft;
    var mouseY = event.clientY - canvas.offsetTop;
    selectedImage = null; // Reset to ensure topmost image is selectedß
    // Create a temporary canvas to test individual images
    var tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    var tempContext = tempCanvas.getContext('2d');

    // Check images in reverse order to prioritize the topmost one
    Object.keys(imagePositions).reverse().forEach(key => {
        var pos = imagePositions[key];
        if (mouseX >= pos.x && mouseX <= pos.x + pos.width && mouseY >= pos.y && mouseY <= pos.y + pos.height) {
            // Clear the temporary canvas and draw only the current image
            tempContext.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
            tempContext.drawImage(images[key], pos.x, pos.y, pos.width, pos.height);

            // Get image data at the cursor position from the temporary canvas
            var imageData = tempContext.getImageData(mouseX, mouseY, 1, 1);
            var data = imageData.data; // [R, G, B, A]

            // Check if the alpha channel is not transparent
            if (data[3] > 0) {
                selectedImage = key;
                dragOffsetX = mouseX - pos.x;
                dragOffsetY = mouseY - pos.y;
                return; // Exit the loop once a non-transparent pixel is found
            }
        }
    });

    // Clean up the temporary canvas
    tempCanvas.remove();
};


canvas.onmousemove = function (event) {
    if (selectedImage) {
        var mouseX = event.clientX - canvas.offsetLeft;
        var mouseY = event.clientY - canvas.offsetTop;
        imagePositions[selectedImage].x = mouseX - dragOffsetX;
        imagePositions[selectedImage].y = mouseY - dragOffsetY;
        drawImages();
    }
};

canvas.onmouseup = function () {
    selectedImage = null;
};

document.getElementById('imageLoader').onchange = function (event) {
    // clear the canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (var key in images) {
        delete images[key];
    }
    for (var key in imagePositions) {
        delete imagePositions[key];
    }
    // clear sketch layers too 
    const layersContainer = document.getElementById('strokeLayersDisplayContainer');
    layersContainer.innerHTML = '';

    var reader = new FileReader();
    reader.onload = function (event) {
        var img = new Image();
        img.onload = function () {
            images['0'] = img;
            imagePositions['0'] = { x: 0, y: 0, width: 512, height: 512 };
            drawImages();
        };
        img.src = event.target.result;

        console.log('image src:', img.src)
    };
    reader.readAsDataURL(event.target.files[0]);
    console.log("file name:", event.target.files[0].name)
    loadedImageName = event.target.files[0].name.split('.')[0]
};

function fetchGetDemoImages(demo) {
    fetch(`/get-images/${demo}`)
        .then(response => response.json())
        .then(imageUrls => {
            console.log('Image URLs:', imageUrls)
            for (var key in images) {
                delete images[key];
            }
            for (var key in imagePositions) {
                delete imagePositions[key];
            }
            loadImages(imageUrls, function (loadedImages) {
                for (var key in loadedImages) {
                    images[key] = loadedImages[key];
                    imagePositions[key] = { x: 0, y: 0, width: 512, height: 512 };
                }
                console.log('Images:', images, imagePositions)

                drawImages();

                const layersContainer = document.getElementById('strokeLayersDisplayContainer');
                layersContainer.innerHTML = '';
                addImagesToDiv(images, layersContainer);
            });


        })
        .catch(error => console.error('Error loading the images:', error));
}
function setupDemoCallback(demos) {
    const container = document.getElementById('buttonContainer');
    if (container) {
        demos.forEach(demo => {
            const button = document.createElement('button');
            button.textContent = `${demo}`;
            button.classList.add('loadDemo');
            button.classList.add('bg-violet-100');
            button.classList.add('text-violet-800');
            button.classList.add('hover:bg-violet-700');
            button.classList.add('hover:text-white');
            button.classList.add('text-sm');
            button.classList.add('p-1');
            button.classList.add('rounded');
            button.setAttribute('data-demo', demo);
            container.appendChild(button);

            button.addEventListener('click', function () {
                fetchGetDemoImages(demo);
            });
        });
    }
}

document.addEventListener('DOMContentLoaded', function () {
    fetch('/get-demos')
        .then(response => response.json())
        .then(demos => {
            setupDemoCallback(demos);
        });
});

function displayImage(imagePath) {
    const container = document.getElementById('imageDisplayContainer');
    const img = new Image(); // Create a new Image element
    img.src = imagePath; // Set the source path
    img.alt = 'Inpainted Image'; // Set an alternative text
    img.onload = function () {
        container.innerHTML = ''; // Clear any previous images/content
        container.appendChild(img); // Add the new image to the container
    };
    img.onerror = function () {
        console.error('Error loading the image.');
    };
}

function addImagesToDiv(images, divElement) {
    var clickCallBack = function (specificKey, specificImgElement) {
        selectedLayer = specificKey;
        console.log('Selected image:', selectedImage)

        // change the border to red for the selected image
        for (var otherkey in images) {
            const otherImg = document.querySelector(`img[src="${images[otherkey].src}"]`);
            otherImg.classList.remove('border-red-500');
        }
        specificImgElement.classList.add('border-red-500');
    }
    Object.keys(imagePositions).reverse().forEach(key => {
        const caption = document.createElement('p');
        reverse_key = Object.keys(imagePositions).length - key - 1;
        caption.textContent = `Layer ${reverse_key}`;
        const img = document.createElement('img');
        img.src = images[key].src;
        img.classList.add('border-2');
        img.classList.add('bg-white')
        const figure = document.createElement('figure');
        figure.appendChild(img);
        figure.appendChild(caption);
        divElement.appendChild(figure);

        // add callback so when we click on the layer, it becomes the selected one

        img.addEventListener('click', clickCallBack.bind(null, key, img));
    });
    console.log('Images:', images)
}

// Add this to your existing custom.js or wherever your JavaScript code is maintained
document.getElementById('inpaintButton').addEventListener('click', function () {

    function callBack() {
        if (!hasInpainted) {
            var layerData = [];
            Object.keys(imagePositions).forEach(key => {
                var pos = imagePositions[key];
                layerData.push({
                    image_src: images[key].src,
                    layerId: key,
                    x: pos.x,
                    y: pos.y,
                    width: pos.width,
                    height: pos.height
                });
            });
            hasInpainted = true
            priorToInpaintingLayerData = layerData
        } else {
            layerData = priorToInpaintingLayerData
        }

        fetch('/inpaint-layers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ layers: layerData }) //, holes: holes })
        })
            .then(response => response.json())
            .then(data => {
                console.log('Inpainting successful:', data);
                // draw on canvas
                images = {}
                imagePositions = {}
                images['0'] = new Image();
                images['0'].src = data['inpainted_path'];
                imagePositions['0'] = { x: 0, y: 0, width: 512, height: 512 };

                images['0'].onload = function () {
                    drawImages();
                }

                // display mask
                inpaintMaskContainer = document.getElementById('inpaintMaskContainer');
                inpaintMaskContainer.innerHTML = '';
                mask_img = new Image();
                mask_img.src = data['mask_path'];
                mask_img.onload = function () {
                    inpaintMaskContainer.appendChild(mask_img);
                }
            })
            .catch(error => {
                console.error('Error during inpainting:', error);
            });
    }
    saveCanvasAsBW(callBack);


});

function saveCanvasAsBW(callBack) {
    var canvas = document.getElementById('myCanvas');
    var context = canvas.getContext('2d');

    // Get the image data from the canvas
    var imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    var data = imageData.data;

    // for all transparent pixels, set them to white
    for (var i = 0; i < data.length; i += 4) {
        if (data[i + 3] == 0) {
            data[i] = 255;     // Red
            data[i + 1] = 255; // Green
            data[i + 2] = 255; // Blue
            data[i + 3] = 255; // Alpha
        }
    }


    // Convert to black and white
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        if (avg > 200) {
            avg = 255;
        } else {
            avg = 0;
        }
        data[i] = avg;     // Red
        data[i + 1] = avg; // Green
        data[i + 2] = avg; // Blue
    }

    // Put the modified data back on the canvas
    context.putImageData(imageData, 0, 0);

    // Convert canvas to data URL
    var dataURL = canvas.toDataURL('image/png');

    console.log(images)
    console.log(images['0'].src.split('/'))
    var sketch_links = images['0'].src.split('/')
    var sketch_name = sketch_links[sketch_links.length - 3];
    console.log('Sketch name:', sketch_name)

    // Send the B&W image to the server
    fetch('/save-bw-image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL, name: sketch_name })
    })
        .then(response => response.json())
        .then(data => {
            console.log('Save successful:', data);
            callBack();
        })
        .catch(error => {
            console.error('Error saving the image:', error);
        });
}


document.getElementById('segmentButton').addEventListener('click', function () {
    // dummy right now, load demo with the same name as the sketch
    fetchGetDemoImages(loadedImageName);
})

document.getElementById('deleteLayerButton').addEventListener('click', function () {
    if (selectedLayer) {
        delete images[selectedLayer];
        delete imagePositions[selectedLayer];
        // we get negative keys, so we need to reassign the keys
        var newImages = {}
        var newImagePositions = {}
        var counter = 0;
        for (var key in images) {
            newImages[counter.toString()] = images[key];
            newImagePositions[counter.toString()] = imagePositions[key];
            counter++;
        }
        images = newImages;
        imagePositions = newImagePositions;
        drawImages();
        const layersContainer = document.getElementById('strokeLayersDisplayContainer');
        layersContainer.innerHTML = '';
        addImagesToDiv(images, layersContainer);
    }
})

document.getElementById('copyLayerButton').addEventListener('click', function () {
    if (selectedLayer) {
        var newKey = Object.keys(images).length.toString();
        images[newKey] = images[selectedLayer];
        imagePositions[newKey] = { ...imagePositions[selectedLayer] };
        drawImages();
        const layersContainer = document.getElementById('strokeLayersDisplayContainer');
        layersContainer.innerHTML = '';
        addImagesToDiv(images, layersContainer);
    }
})