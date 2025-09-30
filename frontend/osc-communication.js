var port = new osc.WebSocketPort({
    url: "ws://localhost:8081"
});

/*
let MEANDERS_LIST = [];
let newMeanderIndices = undefined;
function callDrawPointAtWithRetry(x, y, z, tries = 30, delayMs = 100) {
    if (typeof drawPointAt === 'function') {
        try { drawPointAt(x, y, z); } catch (e) { console.warn('drawPointAt failed:', e); }
        return;
    }
    if (tries > 0) {
        setTimeout(() => callDrawPointAtWithRetry(x, y, z, tries - 1, delayMs), delayMs);
    } else {
        console.warn('drawPointAt not available after retries; skipping point draw');
    }
}

port.on("message", function (oscMessage) {
    $("#message").text(JSON.stringify(oscMessage, undefined, 2));
    MEANDERS_LIST.push(oscMessage.args[0].split("-"));
    newMeanderIndices = oscMessage.args[0].split("-");
    //console.log(newMeanderIndices);
    //console.log("message", oscMessage.args[0].split("-"));
    //console.log("message", oscMessage[0].split("-"));
});
*/

// WebSocket connection debugging
port.on("open", function () {
    console.log("✅ WebSocket connection opened successfully!");
});

port.on("error", function (error) {
    console.error("❌ WebSocket connection error:", error);
});

port.on("close", function () {
    console.log("🔌 WebSocket connection closed");
});

console.log("🔄 Attempting to connect to WebSocket at ws://localhost:8081");
port.open();

// Helper to ensure drawPointAt is called even if main.js isn't ready yet
function callDrawPointAtWithRetry(x, y, z, tries = 30, delayMs = 100) {
    if (typeof drawPointAt === 'function') {
        try { drawPointAt(x, y, z); } catch (e) { console.warn('drawPointAt failed:', e); }
        return;
    }
    if (tries > 0) {
        setTimeout(() => callDrawPointAtWithRetry(x, y, z, tries - 1, delayMs), delayMs);
    } else {
        console.warn('drawPointAt not available after retries; skipping point draw');
    }
}

// Listen for incoming messages from Node.js server
port.on("message", function (oscMessage) {
    console.log("Received OSC message:", oscMessage);
    console.log("Message address:", oscMessage.address, "Type:", typeof oscMessage.address);
    console.log("Address comparison:", oscMessage.address === "/drawBox");
    
    // Handle drawBox messages
    if (oscMessage.address === "/drawBox") {
        // The args are coming as plain values, not OSC-formatted objects
        const x = oscMessage.args[0];
        const y = oscMessage.args[1];
        const z = oscMessage.args[2];
        const colorHue = oscMessage.args[3];
        const arrayIndex = oscMessage.args[4];
        const prevElapsedSec = oscMessage.args[5];
        
        console.log(`Calling drawBox with x=${x}, y=${y}, z=${z}, colorHue=${colorHue}, arrayIndex=${arrayIndex}`);
        
        // Call the drawBox function in main.js, passing prevElapsedSec (seconds) for previous box duration
        drawBox(x, y, z, colorHue, arrayIndex, prevElapsedSec);
        // Also draw a point at the same coordinates in the 3D scene (retry if main.js not ready yet)
        callDrawPointAtWithRetry(x, y, z);
    } else if (oscMessage.address === "/drawCrossfade") {
        console.log("Received drawCrossfade message");
        drawCrossfade();
    } else if (oscMessage.address === "/drawMeander") {
        console.log("Received drawMeander message");
        drawMeander();
    } else {
        console.log("Address did not match with any handler" + oscMessage.address);
    }
});

// get x, y, z coordinates and play corresponding sound
var sendBox = function (send_x, send_y, send_z){
    port.send({
        address: "/play/box",
        args: [
            {
                type: "f",
                value: send_x
            },
            {
                type: "f",
                value: send_y
            },
            {
                type: "f",
                value: send_z
            }
        ]
    });
}

var sendMeander = function (send_start_x, send_start_y, send_start_z, send_end_x, send_end_y, send_end_z, meander_time){
    port.send({
        address: "/play/meander",
        args: [
            {
                type: "f",
                value: send_start_x
            },
            {
                type: "f",
                value: send_start_y
            },
            {
                type: "f",
                value: send_start_z
            },
            {
                type: "f",
                value: send_end_x
            },
            {
                type: "f",
                value: send_end_y
            },
            {
                type: "f",
                value: send_end_z
            },
            {
                type: "f",
                value: meander_time
            }

        ]
    });
}

var sendDrawMeander = function (send_start_x, send_start_y, send_start_z, send_end_x, send_end_y, send_end_z){
    port.send({
        address: "/draw/meander",
        args: [
            {
                type: "f",
                value: send_start_x
            },
            {
                type: "f",
                value: send_start_y
            },
            {
                type: "f",
                value: send_start_z
            },
            {
                type: "f",
                value: send_end_x
            },
            {
                type: "f",
                value: send_end_y
            },
            {
                type: "f",
                value: send_end_z
            },
        ]
    });
}

var sendCrossfade = function (send_start_x, send_start_y, send_start_z, send_end_x, send_end_y, send_end_z, meander_time){
    port.send({
        address: "/play/crossfade",
        args: [
            {
                type: "f",
                value: send_start_x
            },
            {
                type: "f",
                value: send_start_y
            },
            {
                type: "f",
                value: send_start_z
            },
            {
                type: "f",
                value: send_end_x
            },
            {
                type: "f",
                value: send_end_y
            },
            {
                type: "f",
                value: send_end_z
            },
            {
                type: "f",
                value: meander_time
            }

        ]
    });
}

var sendStop = function (){
    port.send({
        address: "/stop",
        args: []
    });
}

var sendStartrecording = function (){
    port.send({
        address: "/startrecording",
        args: []
    });
}

var sendStoprecording = function (){
    port.send({
        address: "/stoprecording",
        args: []
    });
}
