import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@5/+esm";
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import calculateCurrentCompostionTime from './main.js';

// VISUALIZATION PROPERTIES
const scale_x = 100;
const scale_y = 200;
const scale_z = 300;

const BASE_OPACITY = 0.7;


// DATA
const x = new Float32Array(dataset3D_withcolors['x']); //.slice(0, 100);
const y = new Float32Array(dataset3D_withcolors['y']); //.slice(0, 100);
const z = new Float32Array(dataset3D_withcolors['z']); //.slice(0, 100);
const r = new Float32Array(dataset3D_withcolors['r']); //.slice(0, 100);
const g = new Float32Array(dataset3D_withcolors['g']); //.slice(0, 100);
const b = new Float32Array(dataset3D_withcolors['b']); //.slice(0, 100);
const N_POINTS = x.length;
let particles;

let renderer, scene, camera, material, controls, stats;
let raycaster, intersects;
let pointer, INTERSECTED;
var SELECTED_ELEMENT = null;
let canClick = false;
const pickPosition = {x: 0, y: 0};
let MOUSEONSCATTERPLOT = false;

const PARTICLE_SIZE = 15;

init();

function init() {

    // SCENE
    scene = new THREE.Scene();

    // CAMERA
    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight , 1, 10000 );
    camera.position.z = 1500;

    // RENDERER
    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.setAnimationLoop( animate );
    document.getElementById( 'scatterPlot' ).prepend( renderer.domElement );

    // ORBIT CONTROLS
    controls = new OrbitControls( camera, renderer.domElement );
    controls.listenToKeyEvents( window ); // optional
    //controls.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)
    controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 1;
    controls.maxDistance = 1500;
    controls.maxPolarAngle = Math.PI / 2;

    // RENDERING POINTS AS CIRCLES
	//const sprite = new THREE.TextureLoader().load( 'imgs/disc.png' );
	//sprite.colorSpace = THREE.SRGBColorSpace;

    // LOAD DATASETS
    const colors = [];
    const sizes = new Float32Array( N_POINTS );
    const color = new THREE.Color();
	const geometry = new THREE.BufferGeometry();
	const vertices = [];
    const opacities = new Float32Array( N_POINTS );
	for ( let i = 0; i < N_POINTS; i ++ ) {
        let this_x = x[i] * scale_x - (scale_x/2);
        let this_y = y[i] * scale_y - (scale_y/2);
        let this_z = z[i] * scale_z - (scale_z/2);
		vertices.push( this_x, this_y, this_z);

        //const vx = Math.random();
        //const vy = Math.random();
        //const vz = Math.random();
        //color.setRGB( vx, vy, vz );
        color.setRGB( 255, 0, 0 );

        colors.push( color.r, color.g, color.b );
        sizes[i] = PARTICLE_SIZE;
        opacities[i] = BASE_OPACITY;
	}
	geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );
    geometry.setAttribute( 'customColor', new THREE.Float32BufferAttribute( colors, 3 ) );
    geometry.setAttribute( 'size', new THREE.Float32BufferAttribute( sizes, 1 ) );
    geometry.setAttribute( 'opacity', new THREE.Float32BufferAttribute( opacities, 1 ) );

    //material = new THREE.PointsMaterial( { size: 0.05, vertexColors: true, map: sprite } );
    // GEOMETRY MATERIAL
    material = new THREE.ShaderMaterial( {
        uniforms: {
            color: { value: new THREE.Color( 0xffffff ) },
            pointTexture: { value: new THREE.TextureLoader().load( 'imgs/disc.png' ) },
            alphaTest: { value: 0.9 }
        },
        vertexShader: document.getElementById( 'vertexshader' ).textContent,
        fragmentShader: document.getElementById( 'fragmentshader' ).textContent,
        blending: THREE.AdditiveBlending,
        depthTest: false,
        transparent: true
    } );

	// RENDER POINTS
	particles = new THREE.Points( geometry, material );
	scene.add( particles );

    // CLICK INTERACTION
    raycaster = new THREE.Raycaster();
    pointer = new THREE.Vector2();
    window.addEventListener( 'resize', onWindowResize );
    document.addEventListener( 'pointermove', onPointerMove );

    // STATS
    stats = new Stats();
    //document.getElementById( 'scatterPlot' ).appendChild( stats.dom );

}

// UPDATE POINTER POSITION
function onPointerMove( event ) {
    pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
}
// UPDATE WINDOW SIZE
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth-10 , window.innerHeight-10 );
}
// ANIMATE FOR CAMERA NAVIGATION
function animate() {
    //controls.update(); // only required if controls.enableDamping = true, or if controls.autoRotate = true
    render();
}
// RENDER FUNCTION FOR ANIMATION
function render() {
    const time = Date.now() * 0.5;

    if ( SELECTED_ELEMENT == null ){
	    pickHelper.pick(pickPosition, scene, camera, time);
	    if ( canClick ){ // check for click and not drag
	        pickHelper.click(pickPosition, scene, camera, time);
	        canClick = false;
	    }
    }
    // PICKED INDEX DISAPPEARS WHEN OUT OF SCATTERPLOT
    if ( !MOUSEONSCATTERPLOT ){
        pointToBasic(CURRENTPICKEDINDEX);
        CURRENTPICKEDINDEX = null;
    }
    renderer.render( scene, camera );
}

let CURRENTPICKEDINDEX = null;

// HANDLE PICK AND CLICK EVENTS
let clickedIndices = [];
let canvas = renderer.domElement;
class PickHelper {
    constructor() {
      this.raycaster = new THREE.Raycaster();
      this.pickedObject = null;
      this.pickedObjectIndex = null;
      this.pickedObjectSavedColor = 0;
      this.clickedObject = null;
      this.clickedObjectIndex = null;
    }
    pick(normalizedPosition, scene, camera, time) {
        // restore the color if there is a picked object
        if ( !clickedIndices.includes(CURRENTPICKEDINDEX) ) {
            pointToBasic(CURRENTPICKEDINDEX);
            this.pickedObject = undefined;
            this.pickedObjectIndex = undefined;
        }
        
        /*if (this.pickedObject) {
            if ( !clickedIndices.includes(this.pickedObjectIndex) ) {
                particles.geometry.attributes.size.array[ this.pickedObjectIndex ] = PARTICLE_SIZE;
                particles.geometry.attributes.opacity.array[ this.clickedObjectIndex ] = BASE_OPACITY;
                let newcolor = new THREE.Color();
                newcolor.setRGB( 255, 0, 0 );
                particles.geometry.attributes.customColor.array[ this.pickedObjectIndex * 3 ] = newcolor.r;
                particles.geometry.attributes.customColor.array[ this.pickedObjectIndex * 3 + 1 ] = newcolor.g;
                particles.geometry.attributes.customColor.array[ this.pickedObjectIndex * 3 + 2 ] = newcolor.b;
            }
            this.pickedObject = undefined;
            this.pickedObjectIndex = undefined;
        }*/
        // cast a ray through the frustum
        this.raycaster.setFromCamera(normalizedPosition, camera);
        // get the list of objects the ray intersected
        const intersectedObjects = this.raycaster.intersectObjects(scene.children);

        if (intersectedObjects.length) {
            // pick the first object. It's the closest one
            this.pickedObject = intersectedObjects[0].object;
            this.pickedObjectIndex = intersectedObjects[0].index;
            if ( !clickedIndices.includes(this.pickedObjectIndex) ){
                particles.geometry.attributes.size.array[ this.pickedObjectIndex ] = PARTICLE_SIZE * 20;
                particles.geometry.attributes.size.needsUpdate = true;
                // update opacity
                particles.geometry.attributes.opacity.array[ this.clickedObjectIndex ] = 1;
                particles.geometry.attributes.opacity.needsUpdate = true;                
                // change color of picked object to white
                let newcolor = new THREE.Color();
                newcolor.setRGB( 255, 255, 255 );
                particles.geometry.attributes.customColor.array[ this.pickedObjectIndex * 3 ] = newcolor.r;
                particles.geometry.attributes.customColor.array[ this.pickedObjectIndex * 3 + 1 ] = newcolor.g;
                particles.geometry.attributes.customColor.array[ this.pickedObjectIndex * 3 + 2 ] = newcolor.b;
                particles.geometry.attributes.customColor.needsUpdate = true;
                //material.needsUpdate = true
            }
            //console.log("picked ID: "+intersectedObjects[0].index);
            sendBox(x[this.pickedObjectIndex], y[this.pickedObjectIndex], z[this.pickedObjectIndex], this.pickedObjectIndex);
        } else {
            sendStop();
        }
        CURRENTPICKEDINDEX = this.pickedObjectIndex;
    }
    click(normalizedPosition, scene, camera, time) {
        // restore the color if there is a picked object
        if (this.clickedObject) {
            this.clickedObject = undefined;
            this.clickedObjectIndex = undefined;
        }
        // cast a ray through the frustum
        this.raycaster.setFromCamera(normalizedPosition, camera);
        // get the list of objects the ray intersected
        const intersectedObjects = this.raycaster.intersectObjects(scene.children);
        if (intersectedObjects.length) {
            if ( intersectedObjects[0].index != this.clickedObjectIndex ){
                let compositionTime = calculateCurrentCompostionTime();
                if ( compositionTime < MAX_COMPOSITION_DURATION){
                    
                    // click the first object. It's the closest one            
                    this.clickedObject = intersectedObjects[0].object;
                    this.clickedObjectIndex = intersectedObjects[0].index;
                    clickedIndices.push(this.clickedObjectIndex);
                    // update size
                    particles.geometry.attributes.size.array[ this.clickedObjectIndex ] = PARTICLE_SIZE * 20;
                    particles.geometry.attributes.size.needsUpdate = true;
                    // update opacity
                    particles.geometry.attributes.opacity.array[ this.clickedObjectIndex ] = 1;
                    particles.geometry.attributes.opacity.needsUpdate = true;
                    // update color
                    let newcolor = new THREE.Color();
                    let newHueValue = Math.random();
                    let newRGBvalues = colorHsbToRgb( newHueValue*360, 0.9*100, 0.9*100 );
                    newcolor.setRGB( newRGBvalues[0]/255, newRGBvalues[1]/255, newRGBvalues[2]/255 );
                    particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 ] = newcolor.r;
                    particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 + 1 ] = newcolor.g;
                    particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 + 2 ] = newcolor.b;
                    particles.geometry.attributes.customColor.needsUpdate = true;
                    material.needsUpdate = true;
                    console.log("clicked ID: "+intersectedObjects[0].index);

                    drawBox(x[ this.clickedObjectIndex ], y[ this.clickedObjectIndex ], z[ this.clickedObjectIndex ], 
                        newHueValue, this.clickedObjectIndex); 
                }
            }
        }
    }
}

export default function changePointSize ( box_n, size ){
    particles.geometry.attributes.size.array[ compositionArray[box_n].arrayIndex ] = PARTICLE_SIZE * size * heightToPointSize;
    particles.geometry.attributes.size.needsUpdate = true;
}



const colorHsbToRgb = (h, s, b) => {
    s /= 100;
    b /= 100;
    const k = (n) => (n + h / 60) % 6;
    const f = (n) => b * (1 - s * Math.max(0, Math.min(k(n), 4 - k(n), 1)));
    return [255 * f(5), 255 * f(3), 255 * f(1)];
};

clearPickPosition();
function getCanvasRelativePosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * canvas.width  / rect.width,
        y: (event.clientY - rect.top ) * canvas.height / rect.height,
    };
}
function setPickPosition(event) {
    const pos = getCanvasRelativePosition(event);
    pickPosition.x = (pos.x / canvas.width ) *  2 - 1;
    pickPosition.y = (pos.y / canvas.height) * -2 + 1;  // note we flip Y
}
function clearPickPosition() {
    // unlike the mouse which always has a position
    // if the user stops touching the screen we want
    // to stop picking. For now we just pick a value
    // unlikely to pick something
    pickPosition.x = -100000;
    pickPosition.y = -100000;
}
window.addEventListener('mousemove', setPickPosition);
window.addEventListener('mouseout', clearPickPosition);
window.addEventListener('mouseleave', clearPickPosition);
window.addEventListener('touchstart', (event) => {
    // prevent the window from scrolling
    event.preventDefault();
    setPickPosition(event.touches[0]);
  }, {passive: false});
window.addEventListener('touchmove', (event) => {
    setPickPosition(event.touches[0]);
});
window.addEventListener('touchend', clearPickPosition);

const pickHelper = new PickHelper();
let isMouseDown = false;
let timer = 0;
let startTime = 0;
let endTime = 0;
canvas.onmousedown = function(){
    isMouseDown = true;
    startTime = new Date().getTime();
    let timer = 0;
}; 
canvas.onmouseup = function(){
    isMouseDown = false;
    endTime = new Date().getTime();
    timer = endTime -startTime;
}; 

document.getElementById("scatterPlot").addEventListener("mouseenter", function(  ) {
    MOUSEONSCATTERPLOT=true;
});
document.getElementById("scatterPlot").addEventListener("mouseout", function(  ) { 
    MOUSEONSCATTERPLOT=false;
});

// RENDERING FUNCTIONS
function pointToBasic(pointIndex){
    // restore point rendering to the basic properties
    // update size
    particles.geometry.attributes.size.array[ pointIndex ] = PARTICLE_SIZE;
    particles.geometry.attributes.size.needsUpdate = true;
    // update color
    let newcolor = new THREE.Color();
    newcolor.setRGB( 255, 0, 0 );
    particles.geometry.attributes.customColor.array[ pointIndex * 3 ] = newcolor.r;
    particles.geometry.attributes.customColor.array[ pointIndex * 3 + 1 ] = newcolor.g;
    particles.geometry.attributes.customColor.array[ pointIndex * 3 + 2 ] = newcolor.b;
    particles.geometry.attributes.customColor.needsUpdate = true;
    material.needsUpdate = true
}

// Function to add a new point to the scene dynamically
window.addNewPointToScene = function addNewPointToScene(x_coord, y_coord, z_coord) {
    console.log(`Adding new point to scene: (${x_coord}, ${y_coord}, ${z_coord})`);
    
    // Add to underlying data arrays (important for index synchronization)
    const newIndex = x.length;
    
    // Extend the data arrays by creating new ones
    const newX = new Float32Array(x.length + 1);
    const newY = new Float32Array(y.length + 1);
    const newZ = new Float32Array(z.length + 1);
    const newR = new Float32Array(r.length + 1);
    const newG = new Float32Array(g.length + 1);
    const newB = new Float32Array(b.length + 1);
    
    // Copy existing data
    newX.set(x);
    newY.set(y);
    newZ.set(z);
    newR.set(r);
    newG.set(g);
    newB.set(b);
    
    // Derive RGB color from parameters (normalize from 0-127 to 0-1)
    // Use first 3 parameters to determine RGB
    let newR_val, newG_val, newB_val;
    if (parameters && parameters.length >= 3) {
        newR_val = parameters[0] / 127.0;
        newG_val = parameters[1] / 127.0;
        newB_val = parameters[2] / 127.0;
    } else {
        // Fallback to green if no parameters
        newR_val = 0;
        newG_val = 1;
        newB_val = 0;
    }
    
    // Add new point to data arrays
    newX[newIndex] = x_coord;
    newY[newIndex] = y_coord;
    newZ[newIndex] = z_coord;
    newR[newIndex] = 0; // Green color for new points
    newG[newIndex] = 1;
    newB[newIndex] = 0;
    
    // Update the global arrays (reassign to window to make them accessible)
    window.x = newX;
    window.y = newY;
    window.z = newZ;
    window.r = newR;
    window.g = newG;
    window.b = newB;
    
    console.log(`Updated data arrays. New length: ${newX.length}, new index: ${newIndex}`);
    
    // Transform coordinates to scene space
    const this_x = x_coord * scale_x - (scale_x/2);
    const this_y = y_coord * scale_y - (scale_y/2);
    const this_z = z_coord * scale_z - (scale_z/2);
    
    // Get current geometry attributes
    const positions = particles.geometry.attributes.position;
    const colors = particles.geometry.attributes.customColor;
    const sizes = particles.geometry.attributes.size;
    const opacities = particles.geometry.attributes.opacity;
    
    // Create new arrays with one more element
    const newPositions = new Float32Array(positions.count * 3 + 3);
    const newColors = new Float32Array(colors.count * 3 + 3);
    const newSizes = new Float32Array(sizes.count + 1);
    const newOpacities = new Float32Array(opacities.count + 1);
    
    // Copy existing data
    newPositions.set(positions.array);
    newColors.set(colors.array);
    newSizes.set(sizes.array);
    newOpacities.set(opacities.array);
    
    // Add new point data
    const newIndex = positions.count;
    newPositions[newIndex * 3] = this_x;
    newPositions[newIndex * 3 + 1] = this_y;
    newPositions[newIndex * 3 + 2] = this_z;
    
    // Set color (green for new points)
    newColors[newIndex * 3] = 0;
    newColors[newIndex * 3 + 1] = 1;
    newColors[newIndex * 3 + 2] = 0;
    
    // Set size and opacity
    newSizes[newIndex] = PARTICLE_SIZE * 1.5; // Make new points slightly larger
    newOpacities[newIndex] = 1.0; // Full opacity for new points
    
    // Update geometry attributes
    particles.geometry.setAttribute('position', new THREE.Float32BufferAttribute(newPositions, 3));
    particles.geometry.setAttribute('customColor', new THREE.Float32BufferAttribute(newColors, 3));
    particles.geometry.setAttribute('size', new THREE.Float32BufferAttribute(newSizes, 1));
    particles.geometry.setAttribute('opacity', new THREE.Float32BufferAttribute(newOpacities, 1));
    
    // Mark for update
    particles.geometry.attributes.position.needsUpdate = true;
    particles.geometry.attributes.customColor.needsUpdate = true;
    particles.geometry.attributes.size.needsUpdate = true;
    particles.geometry.attributes.opacity.needsUpdate = true;
    
    console.log(`New point added. Total points: ${newIndex + 1}`);
}




