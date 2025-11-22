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
    // Send stop signal on page load/refresh
    console.log("🛑 Sending stop signal on page load");
    sendStop();
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
        const durationMs = oscMessage.args.length > 0 ? oscMessage.args[0] : null;
        console.log("Received drawCrossfade message", durationMs ? `with duration ${durationMs}ms` : "");
        drawCrossfade(durationMs);
    } else if (oscMessage.address === "/drawMeander") {
        const durationMs = oscMessage.args.length > 0 ? oscMessage.args[0] : null;
        console.log("Received drawMeander message", durationMs ? `with duration ${durationMs}ms` : "");
        drawMeander(durationMs);
    } else if (oscMessage.address === "/newPoint") {
        // Handle new point addition from training mode
        try {
            const jsonData = JSON.parse(oscMessage.args[0]);
            const x = jsonData.x;
            const y = jsonData.y;
            const z = jsonData.z;
            const parameters = jsonData.parameters;
            
            console.log(`Received new point: (${x}, ${y}, ${z}) with params:`, parameters);
            
            // Add point to scene (function is available on window from scatterplot.js)
            if (typeof window.addNewPointToScene === 'function') {
                const newPointIndex = window.addNewPointToScene(x, y, z, parameters);
                console.log('New point added to 3D visualization at index:', newPointIndex);
                
                // Wait for next animation frame to ensure geometry is fully updated
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        // After adding to scene, draw the box in the UI
                        const colorHue = Math.floor(Math.random() * 7) + 3; // Random color 3-9
                        const prevElapsedSec = -1; // No previous elapsed time for first new point
                        
                        console.log(`Drawing box for new point: x=${x}, y=${y}, z=${z}, index=${newPointIndex}`);
                        drawBox(x, y, z, colorHue, newPointIndex, prevElapsedSec);
                    });
                }); // Double RAF ensures render is complete.
            } else {
                console.error('addNewPointToScene function not available');
            }
        } catch (e) {
            console.error('Error parsing newPoint message:', e);
        }
    } else {
        console.log("Address did not match with any handler" + oscMessage.address);
    }
});

// Global playback state flag
var IS_PLAY_ON = false;

// get x, y, z coordinates and play corresponding sound
var sendBox = function (send_x, send_y, send_z, send_index){
    IS_PLAY_ON = true;
    console.log(`🎵 PLAY BOX - Coordinates: x=${send_x}, y=${send_y}, z=${send_z}, index=${send_index}`);
    console.log(`   Sending to Pure Data via /play/box`);
    
    const args = [
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
    ];
    
    // Add index if provided
    if (send_index !== undefined && send_index !== null) {
        args.push({
            type: "i",
            value: send_index
        });
    }
    
    port.send({
        address: "/play/box",
        args: args
    });
}

var sendMeander = function (send_start_x, send_start_y, send_start_z, send_end_x, send_end_y, send_end_z, meander_time){
    IS_PLAY_ON = true;
    console.log(`🎵 PLAY MEANDER - Start: (${send_start_x}, ${send_start_y}, ${send_start_z}) → End: (${send_end_x}, ${send_end_y}, ${send_end_z})`);
    console.log(`   Duration: ${meander_time}ms, Sending to Pure Data via /play/meander`);
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
    console.log(`🎵 PLAY CROSSFADE - Start: (${send_start_x}, ${send_start_y}, ${send_start_z}) → End: (${send_end_x}, ${send_end_y}, ${send_end_z})`);
    console.log(`   Duration: ${meander_time}ms, Sending to Pure Data via /play/crossfade`);
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
    IS_PLAY_ON = false;
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
