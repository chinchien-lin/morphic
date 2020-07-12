import * as THREE from 'https://threejs.org/build/three.module.js';

function Mesh() {
    this.nodes = {};
    this.elements = {};
}

function Node(id, x) {
    this.id = id;
    this.x = x;
    this.elements = [];
}

function Element(id, basis, nodes) {
    this.id = id;
    this.basis = basis;
    this.nodes = nodes;
}

Mesh.prototype.addNode = function(id, x) {
    this.nodes[id] = new Node(id, x);
};

Mesh.prototype.addElement = function(id, basis, nodeIds) {
    let nodes = [];
    for (let nid in nodeIds) {
        let enid = nodeIds[nid];
        nodes.push(this.nodes[enid]);
    }
    let element = new Element(id, basis, nodes);
    this.elements[id] = element;
    for (let idx in element.nodes) {
        element.nodes[idx].elements.push(element);
    }
};

Mesh.prototype.loadMesh = function(json) {
    for (let nid in json.nodes) {
        this.addNode(nid, json.nodes[nid]);
    }
    for (var eid in json.elements) {
        let elem = json.elements[eid];
        this.addElement(eid, elem.basis, elem.nodes)
    }
};

Mesh.prototype.evaluate = function(elementId, xi) {
    let w = getWeightsL3L3L3(xi);
    return evaluateElement(this.elements[elementId], w);
};

Mesh.prototype.find = function(point, startingNodeId) {
    let pt = [point.x, point.y, point.z];
    let nodeMaterialPoints = this.getNodeMaterialPoints(startingNodeId);
    for (let i in nodeMaterialPoints) {
        let mpt = nodeMaterialPoints[i];
        let x = this.evaluate(mpt.elementId, mpt.xi)
        let r = calcDistance(x, pt);
        mpt.error = r;
    }
    let t0 = new Date().getTime();
    let n = 0;
    for (let step = 0; step < 10000; step++) {
        for (let i in nodeMaterialPoints) {
            let mpt = nodeMaterialPoints[i];
            let xi = perturbRandom(mpt.xi, 0.1);
            let x = this.evaluate(mpt.elementId, xi);
            let r = calcDistance(x, pt);
            if (r < mpt.error) {
                //console.log(step + "." + i + " " + r + " " + xi + " ************************");
                mpt.error = r;
                mpt.xi = xi;
                if (r < 1.0) {
                    console.log("Time: ("+n+") "+ (new Date().getTime() - t0) + "ms Error: " + r);
                    return mpt;
                }
            }
        }
        n = step;
    }
    let bestIndex = 0;
    let bestError = nodeMaterialPoints[0].error;
    for (let i in nodeMaterialPoints) {
        let mpt = nodeMaterialPoints[i];
        if (mpt.error < bestError) {
            bestIndex = i;
            bestError = mpt.error;
        }
    }
    let mpt = nodeMaterialPoints[bestIndex];
    console.log("Time: ("+n+") "+ (new Date().getTime() - t0) + "ms Error: " + mpt.error);
    return mpt;
};

Mesh.prototype.search = function(point, startingNodeId, tol) {
    let pt = [point.x, point.y, point.z];
    let nodeMaterialPoints = this.getNodeMaterialPoints(startingNodeId);
    for (let i in nodeMaterialPoints) {
        let mpt = nodeMaterialPoints[i];
            //console.log("Material Points to test. Elem: " + mpt.elementId + " xi = " + mpt.xi);
        let x = this.evaluate(mpt.elementId, mpt.xi)
        mpt.error = calcDistance(x, pt);
    }
    let t0 = new Date().getTime();
    let xiStepSize = [0.1, 0.01, 0.001];
    for (let i in nodeMaterialPoints) {
        let mpt = nodeMaterialPoints[i];
        let numDims = mpt.xi.length;
        let nIter = 0;
        let go = true;
        while (go || nIter < 10) {
            let minErr = 1 * mpt.error;
            let minXi;

            //console.log("###################### " + mpt.elementId + " xi = " + mpt.xi + " err = " + mpt.error);
            for (let xiIdx in xiStepSize) {
                let dXi = xiStepSize[xiIdx];
                //console.log(">>>>>>>>>>>>> " + dXi);
                for (let dim = 0; dim < numDims; dim++) {
                    let xi = mpt.xi.slice();
                    //console.log("Testing dim = " + dim + " xi = " + xi);
                    xi[dim] += dXi;
                    if (xi[dim] >= 0 && xi[dim] <= 1) {
                        //console.log("test " + xi);
                        let x = this.evaluate(mpt.elementId, xi)
                        let err = calcDistance(x, pt);
                        if (err < minErr) {
                            minErr = err;
                            minXi = xi;
                            //console.log("Min " + err + " " + xi);
                            if (err < tol) {
                                mpt.error = err;
                                mpt.xi = xi;
                                console.log("Time 1: "+ (new Date().getTime() - t0) + "ms Error: " + mpt.error);
                                return mpt;
                            }
                        }
                    } else {
                        //console.log("outside " + xi);
                    }
                    xi = mpt.xi.slice();
                    xi[dim] -= dXi;
                    if (xi[dim] >= 0 && xi[dim] <= 1) {
                        //console.log("test " + xi);
                        let x = this.evaluate(mpt.elementId, xi)
                        let err = calcDistance(x, pt);
                        if (err < minErr) {
                            minErr = err;
                            minXi = xi;
                            //console.log("Min " + err + " " + xi);
                            if (err < tol) {
                                mpt.error = err;
                                mpt.xi = xi;
                                console.log("Time 2: "+ (new Date().getTime() - t0) + "ms Error: " + mpt.error);
                                return mpt;
                            }
                        }
                    } else {
                        //console.log("outside " + xi);
                    }

                }
            }
            if (minXi == null) {
                go = false;
            } else {
                mpt.xi = minXi;
                mpt.error = minErr;
            }
            nIter++;
        }
    }

    let bestIndex = 0;
    let bestError = nodeMaterialPoints[0].error;
    for (let i in nodeMaterialPoints) {
        let mpt = nodeMaterialPoints[i];
        if (mpt.error < bestError) {
            bestIndex = i;
            bestError = mpt.error;
        }
    }
    let mpt = nodeMaterialPoints[bestIndex];
    console.log("Time 3: "+ (new Date().getTime() - t0) + "ms Error: " + mpt.error);
    return mpt;
};

function calcDistance(x0, x1) {
    let dx2 = 0;
    for (let i = 0; i < x0.length; i++) {
        let dx = x0[i] - x1[i];
        dx2 += dx * dx;
    }
    return Math.sqrt(dx2);
}

function perturbRandom(x, dx) {
    let xp = [];
    for (let i = 0; i < x.length; i++) {
        xp.push(x[i] + 2 * dx * (Math.random() - 0.5));
        if (xp[i] < 0) {
            xp[i] = 0;
        } else if (xp[i] > 1) {
            xp[i] = 1;
        }
    }
    return xp;
}

Mesh.prototype.getNodeMaterialPoints = function(nodeId) {
    let materialPoints = [];
    let node = this.nodes[nodeId];
    for (let eid in node.elements) {
        let element = node.elements[eid];
        let nodeIdx = element.nodes.indexOf(node);
        let xi0 = (nodeIdx % 4) / 3;
        let xi1 = Math.floor((nodeIdx % 16 / 4)) / 3;
        let xi2 = Math.floor(nodeIdx / 16) / 3;
        materialPoints.push({elementId: element.id, xi: [xi0, xi1, xi2]});
    }
    return materialPoints;
};

Mesh.prototype.getNodePositions = function () {
  let nodePositions = [];

  for (const [id, node] of Object.entries(this.nodes)) {
    let nodePosition = new THREE.Vector3();
    if (Array.isArray(node.x[0])) {
      nodePosition.x = node.x[0][0];
      nodePosition.y = node.x[1][0];
      nodePosition.z = node.x[2][0];
    } else {
      nodePosition.x = node.x[0];
      nodePosition.y = node.x[1];
      nodePosition.z = node.x[2];
    }
    nodePositions.push(nodePosition);
  }
  return nodePositions;
};

Mesh.prototype.getNodeids = function() {
    let nodeIds = [];
    for (const [id, node] of Object.entries(this.nodes)) {
        nodeIds.push(node.id);
    }
    return nodeIds;
};

function evaluateFaceNodes(basis) {

  // Determine spacing of nodes for a given basis function e.g. basis = 'L1'.
  let nodesPositionsPerXi = [];
  if (basis === 'L1') {
    nodesPositionsPerXi = [0, 1.0];
  } else if (basis === 'L2') {
    nodesPositionsPerXi = [0, 0.5, 1.0];
  } else if (basis === 'L3') {
    nodesPositionsPerXi = [0, 1 / 3, 2 / 3, 1.0];
  }

  // Create a unit cube following OpenCMISS FEM node numbering.
  let nodePositions = [];
  for (const zValue of nodesPositionsPerXi) {
    for (const yValue of nodesPositionsPerXi) {
      for (const xValue of nodesPositionsPerXi) {
        let nodePosition = new THREE.Vector3();
        nodePosition.x = xValue;
        nodePosition.y = yValue;
        nodePosition.z = zValue;
        nodePositions.push(nodePosition);
      }
    }
  }

  // Initialise array for storing node indices relating to each face.
  let faceNodesIdxs = {
    'xi1=0': [],
    'xi1=1': [],
    'xi2=0': [],
    'xi2=1': [],
    'xi3=0': [],
    'xi3=1': []
  };

  // Evaluate the node indices that are part of each face.
  for (const [nodeIdx, nodePosition] of nodePositions.entries()) {
    if (nodePosition.x === 0) {
      faceNodesIdxs['xi1=0'].push(nodeIdx);
    }
    if (nodePosition.x === 1) {
      faceNodesIdxs['xi1=1'].push(nodeIdx);
    }
    if (nodePosition.y === 0) {
      faceNodesIdxs['xi2=0'].push(nodeIdx);
    }
    if (nodePosition.y === 1) {
      faceNodesIdxs['xi2=1'].push(nodeIdx);
    }
    if (nodePosition.z === 0) {
      faceNodesIdxs['xi3=0'].push(nodeIdx);
    }
    if (nodePosition.z === 1) {
      faceNodesIdxs['xi3=1'].push(nodeIdx);
    }
  }
  return faceNodesIdxs;
}

function getMeshLineBasis(mesh) {
  let volumeBasis = Object.values(mesh.elements)[0].basis;
  let lineBasis = volumeBasis[0];
  return lineBasis;
}

function generateSurfaceMesh(mesh) {
  // Determine basis from the first element.
  let volumeBasis = Object.values(mesh.elements)[0].basis;
  let surfBasis = volumeBasis.slice(0, 1 + 1);
  let lineBasis = volumeBasis[0];
  let faceNodesIdxs = evaluateFaceNodes(lineBasis);
  let faces = ['xi1=0', 'xi1=1', 'xi2=0', 'xi2=1', 'xi3=0', 'xi3=1'];
  let surf = new Mesh();
  let elemNum = 0;
  for (const element of Object.values(mesh.elements)) {
    for (const face of faces) {
      let nodeIds = []; // Stores nodes for the new surface element.
      let nodeIdxs = faceNodesIdxs[face];
      for (const nodeIdx of nodeIdxs) {
        let surfNode = element.nodes[nodeIdx];
        nodeIds.push(surfNode.id);
        surf.addNode(surfNode.id, surfNode.x);
      }
      surf.addElement(elemNum, surfBasis, nodeIds);
      elemNum++;
    }
  }
  return surf;
}

function generateSurfaceMesh_old(mesh, faces) {
    let testNodeIdx = [5, 53, 17, 29, 20, 23];
    let faceNodesIdx = {
        5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        53: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
        17: [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51],
        29: [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63],
        20: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
        23: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    };
    var surf = new Mesh();
    for (let eid in mesh.elements) {
        let element = mesh.elements[eid];
        for (let i = 0; i < faces.length; i++) {
            let testIdx = testNodeIdx[faces[i]];
            var testNode = element.nodes[testIdx];
            if (testNode.elements.length == 1) {
                let elementNodes = testNode.elements[0].nodes;
                let faceNodes = faceNodesIdx[testIdx];
                let nodeIds = [];
                for (var j = 0; j < faceNodes.length; j++) {
                    let surfNode = elementNodes[faceNodes[j]];
                    nodeIds.push(surfNode.id);
                    surf.addNode(surfNode.id, surfNode.x);
                }
                surf.addElement(testNode.id, ["L3", "L3"], nodeIds);
            }
        }
    }
    return surf;
}

function addMeshToGeometry(geometry, mesh, resolution) {
  var vertex, vertices, face;
  var res = [resolution, resolution];
  let basis = getMeshLineBasis(mesh);
  var weights = get2DTesselationWeights(res, basis);
  var faces = getTesselationTriangles(res);
  var vertexOffset = 0;
  for (var key in mesh.elements) {
    vertices = tessellateElementL3L3(mesh, key, weights);
    for (var vertIndex in vertices) {
      vertex = vertices[vertIndex];
      geometry.vertices.push(new THREE.Vector3(vertex[0], vertex[1], vertex[2]));
    }
    for (var faceIdex in faces) {
      face = faces[faceIdex];
      geometry.faces.push(new THREE.Face3(face[0] + vertexOffset, face[1] + vertexOffset, face[2] + vertexOffset));
    }
    vertexOffset += vertices.length;
  }
  //geometry.mergeVertices();
  geometry.computeFaceNormals();
  geometry.computeVertexNormals();
  return geometry;
}

function updateGeometryFromMesh(geometry, mesh, resolution) {
  var vertex, elementVertices;
  let vertices = [];
  var res = [resolution, resolution];
  let basis = getMeshLineBasis(mesh);
  var weights = get2DTesselationWeights(res, basis);
  for (var key in mesh.elements) {
    elementVertices = tessellateElementL3L3(mesh, key, weights);
    for (var vertIndex in elementVertices) {
      vertex = elementVertices[vertIndex];
      vertices.push(new THREE.Vector3(vertex[0], vertex[1], vertex[2]));
    }
  }
  // Update existing geometry vertices.
  for (const [vertexIdx, position] of vertices.entries()) {
    geometry.vertices[vertexIdx].setX(position.x);
    geometry.vertices[vertexIdx].setY(position.y);
    geometry.vertices[vertexIdx].setZ(position.z);
  }
  geometry.verticesNeedUpdate;
  geometry.computeFaceNormals();
  geometry.computeVertexNormals;


  return geometry;
}

function updateMesh(mesh, nodeId, nodePosition) {
  if (Array.isArray(mesh.nodes[parseInt(nodeId)].x[0])) {
    mesh.nodes[parseInt(nodeId)].x[0][0] = nodePosition.x
    mesh.nodes[parseInt(nodeId)].x[1][0] = nodePosition.y
    mesh.nodes[parseInt(nodeId)].x[2][0] = nodePosition.z
  } else {
    mesh.nodes[parseInt(nodeId)].x[0] = nodePosition.x
    mesh.nodes[parseInt(nodeId)].x[1] = nodePosition.y
    mesh.nodes[parseInt(nodeId)].x[2] = nodePosition.z
  }
  return mesh
}

function getMeshLines(mesh, resolution) {
    let vertex, vertices;
    let lineIdsDone = [];
    let lines = [];
    let xi0 = [0., 0.];
    let xi1 = [0., 1.];
    let Xi = getXiRange(xi0, xi1, resolution);
    let weights = getWeights(Xi);
    for (let elementId in mesh.elements) {
        let geometry = new THREE.Geometry();
        vertices = evaluateElement(mesh, elementId, weights);
        for (let vertIndex in vertices) {
            vertex = vertices[vertIndex];
            geometry.vertices.push(new THREE.Vector3(vertex[0], vertex[1], vertex[2]));
        }
        lines.push(geometry);
    }
    xi0 = [0., 0.];
    xi1 = [1., 0.];
    Xi = getXiRange(xi0, xi1, resolution);
    weights = getWeights(Xi);
    for (let elementId in mesh.elements) {
        let geometry = new THREE.Geometry();
        vertices = evaluateElement(mesh, elementId, weights);
        for (let vertIndex in vertices) {
            vertex = vertices[vertIndex];
            geometry.vertices.push(new THREE.Vector3(vertex[0], vertex[1], vertex[2]));
        }
        lines.push(geometry);
    }
    return lines;
}

function getXiRange(xi0, xi1, res) {
    var xi;
    var Xi = [];
    var delta = 1 / res;
    for (var i = 0; i < res + 1; i++) {
        xi = i * delta;
        Xi.push([(1- xi) * xi0[0] + xi * xi1[0], (1- xi) * xi0[1] + xi * xi1[1]]);
    }
    return Xi;
}

function evaluateElement(element, weights) {
    let x = [0, 0, 0];
    for (var f = 0; f < 3; f++) {
        for (var i = 0; i < weights.length; i++) {
            x[f] += weights[i] * element.nodes[i].x[f];
        }
    }
    return x;
}

function evaluateElementH3H3(mesh, elementId, weights) {
    var element = mesh.elements[elementId];
    var verts = [];
    for (var widx in weights) {
        verts.push([0, 0, 0]);
    }
    var w;
    var n0 = mesh.nodes[element["nodes"][0]];
    var n1 = mesh.nodes[element["nodes"][1]];
    var n2 = mesh.nodes[element["nodes"][2]];
    var n3 = mesh.nodes[element["nodes"][3]];
    for (var widx in weights) {
        w = weights[widx];
        for (var f = 0; f < 3; f++) {
            verts[widx][f] =
                w[0] * n0[f][0] + w[1] * n0[f][1] + w[2] * n0[f][2] + w[3] * n0[f][3] +
                w[4] * n1[f][0] + w[5] * n1[f][1] + w[6] * n1[f][2] + w[7] * n1[f][3] +
                w[8] * n2[f][0] + w[9] * n2[f][1] + w[10] * n2[f][2] + w[11] * n2[f][3] +
                w[12] * n3[f][0] + w[13] * n3[f][1] + w[14] * n3[f][2] + w[15] * n3[f][3];
        }
    }
    return verts;
}


function tessellateElementL3L3(mesh, elementId, weights) {
    var element = mesh.elements[elementId];
    var verts = [];
    for ( var widx in weights ) {
        verts.push([0, 0, 0]);
    }
    let w;
    for (var f = 0; f < 3; f++) {
        for ( var pt in weights ) {
            w = weights[pt];
            for (var i = 0; i < w.length; i++) {
                verts[pt][f] += w[i] * element.nodes[i].x[f];
            }
        }
    }
    return verts;
}


function tessellateElementH3H3(mesh, elementId, weights) {
    var element = mesh.elements[elementId];
    var verts = [];
    for ( var widx in weights ) {
        verts.push([0, 0, 0]);
    }
    var w, n0, n1, n2, n3;
    for (var f = 0; f < 3; f++) {
        for ( var widx in weights ) {
            w = weights[widx]
            n0 = mesh.nodes[element["nodes"][0]];
            n1 = mesh.nodes[element["nodes"][1]];
            n2 = mesh.nodes[element["nodes"][2]];
            n3 = mesh.nodes[element["nodes"][3]];
            verts[widx][f] =
                w[0] * n0[f][0] + w[1] * n0[f][1] + w[2] * n0[f][2] + w[3] * n0[f][3] +
                w[4] * n1[f][0] + w[5] * n1[f][1] + w[6] * n1[f][2] + w[7] * n1[f][3] +
                w[8] * n2[f][0] + w[9] * n2[f][1] + w[10] * n2[f][2] + w[11] * n2[f][3] +
                w[12] * n3[f][0] + w[13] * n3[f][1] + w[14] * n3[f][2] + w[15] * n3[f][3];
        }
    }
    return verts;
}

function getTesselationTriangles(res) {
    var N0 = res[0] + 1;
    var faces = [];
    var j0, i0;
    for (var j = 0; j < res[1]; j++) {
        j0 = j * N0;
        for (var i = 0; i < res[0]; i++) {
            i0 = j0 + i;
            faces.push([i0, i0 + 1, i0 + N0]);
            faces.push([i0 + N0, i0 + 1, i0 + N0 + 1]);
        }
    }
    return faces;

}

function get2DTesselationWeights(res, basis) {
    var N0 = res[0] + 1;
    var N1 = res[1] + 1;
    var div0 = 1. / res[0];
    var div1 = 1. / res[1];
    if (basis === 'L1') {
      var W0 = Array.apply(null, {length: N0}).map(Number.call, Number).map(function (a) {return div0 * a}).map(L1);
      var W1 = Array.apply(null, {length: N1}).map(Number.call, Number).map(function (a) {return div1 * a}).map(L1);
    }else if (basis === 'L2') {
      var W0 = Array.apply(null, {length: N0}).map(Number.call, Number).map(function (a) {return div0 * a}).map(L2);
      var W1 = Array.apply(null, {length: N1}).map(Number.call, Number).map(function (a) {return div1 * a}).map(L2);
    }else if (basis === 'L3') {
      var W0 = Array.apply(null, {length: N0}).map(Number.call, Number).map(function (a) {return div0 * a}).map(L3);
      var W1 = Array.apply(null, {length: N1}).map(Number.call, Number).map(function (a) {return div1 * a}).map(L3);
    }
    var weights = [];
    for (var j = 0; j < N1; j++) {
        for (var i = 0; i < N0; i++) {
            var w = [];
            for (var k1 in W1[j]) {
                for (var k0 in W0[i]) {
                    w.push(W0[i][k0] * W1[j][k1]);
                }
            }
            weights.push(w);
        }
    }
    return weights;
}

function getTesselationWeightsH3H3(res) {
    var N0 = res[0] + 1;
    var N1 = res[1] + 1;
    var div0 = 1. / res[0];
    var div1 = 1. / res[1];
    var W0 = Array.apply(null, {length: N0}).map(Number.call, Number).map(function(a) {return div0 * a}).map(H3);
    var W1 = Array.apply(null, {length: N1}).map(Number.call, Number).map(function(a) {return div1 * a}).map(H3);
    var weights = [];
    for (var j = 0; j < N1; j++) {
        for (var i = 0; i < N0; i++) {
            var w = [];
            for (var k in mixIdx) {
                w.push(W0[i][mixIdx[k][0]] * W1[j][mixIdx[k][1]])
            }
            weights.push(w);
        }
    }
    return weights;
}

function getWeightsL3L3L3(Xi) {
    let W0, W1, W2;
    W0 = L3(Xi[0]);
    W1 = L3(Xi[1]);
    W2 = L3(Xi[2]);
    let w = [];
    for (var k2 in W2) {
        for (var k1 in W1) {
            for (var k0 in W0) {
                w.push(W0[k0] * W1[k1] * W2[k2]);
            }
        }
    }
    return w;
}

function getWeightsH3H3(Xi) {
    let mixIdx = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [3, 0], [2, 1], [3, 1],
        [0, 2], [1, 2], [0, 3], [1, 3], [2, 2], [3, 2], [2, 3], [3, 3]];
    let weights = [];
    let W0, W1;
    for (var idx in Xi) {
        W0 = H3(Xi[idx][0]);
        W1 = H3(Xi[idx][1]);
        let w = [];
        for (let k in mixIdx) {
            w.push(W0[mixIdx[k][0]] * W1[mixIdx[k][1]]);
        }
        weights.push(w);
    }
    return weights;
}


function getCrossMultiplyIndicies(basis) {

}

// Linear-Lagrange basis function
function L1(x) {
    let L1 = 1 - x;
    let L2 = x;
    return [1. - x, x];
}

// Quadratic-Lagrange basis function
function L2(x) {
    let L1 = 1 - x;
    let L2 = x;
    return [L1 * (2.0 * L1 - 1), 4.0 * L1 * L2, L2 * (2.0 * L2 - 1)];
}

// Cubic-Lagrange basis function
function L3(x) {
    let L1 = 1 - x;
    let L2 = x;
    let sc = 9. / 2.;
    return [0.5*L1*(3*L1-1)*(3*L1-2), sc*L1*L2*(3*L1-1), sc*L1*L2*(3*L2-1), 0.5*L2*(3*L2-1)*(3*L2-2)];
}

// Cubic-Hermite basis function.
function H3(x) {
    let x2 = x * x;
    return [1 - 3 * x2 + 2 * x * x2, x * (x - 1) * (x - 1), x2 * (3 - 2 * x), x2 * (x - 1)];
}

export {Mesh, addMeshToGeometry, generateSurfaceMesh, updateMesh, updateGeometryFromMesh};
