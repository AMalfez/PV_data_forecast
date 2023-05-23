function transpose(array) {
    // Get the number of rows and columns
    const rows = array.length;
    const columns = array[0].length;
  
    // Create a new transposed array
    const transposedArray = [];
    for (let j = 0; j < columns; j++) {
      transposedArray[j] = [];
      for (let i = 0; i < rows; i++) {
        transposedArray[j][i] = array[i][j];
      }
    }
  
    return transposedArray;
  }

  function subtract(array1, array2) {
    // Check if the arrays have the same length
    if (array1.length !== array2.length) {
      throw new Error('Arrays must have the same length.');
    }
  
    // Subtract the elements and create a new array
    const result = [];
    for (let i = 0; i < array1.length; i++) {
      result.push(array1[i] - array2[i]);
    }
  
    return result;
  }
  
  

// ==========================================================================================
function hamm(n, base) {
  let seq = new Array(n).fill(0);

  if (1 < base) {
    let seed = Array.from({ length: n }, (_, i) => i + 1);
    let base_inv = 1 / base;

    while (seed.some((x) => x !== 0)) {
      let digit = seed.map((x) => x % base);
      seq = seq.map((x, i) => x + digit[i] * base_inv);
      base_inv = base_inv / base;
      seed = seed.map((x) => Math.floor(x / base));
    }
  } else {
    let temp = Array.from({ length: n }, (_, i) => i + 1);
    seq = temp.map((x) => ((x % (-base + 1)) + 0.5) / -base);
  }

  return seq;
}

//========================================================================================
function zero_crossings(x) {
  const indzer = [];

  for (let i = 0; i < x.length - 1; i++) {
    if (x[i] * x[i + 1] < 0) {
      indzer.push(i);
    }
  }

  if (x.includes(0)) {
    const iz = [];
    for (let i = 0; i < x.length; i++) {
      if (x[i] === 0) {
        iz.push(i);
      }
    }

    if (iz.length > 1) {
      const diff = [];
      for (let i = 1; i < iz.length; i++) {
        diff.push(iz[i] - iz[i - 1]);
      }

      if (diff.includes(1)) {
        const zer = x.map((value) => value === 0);
        const dz = [];
        for (let i = 0; i < zer.length - 1; i++) {
          dz.push(zer[i + 1] - zer[i]);
        }

        const debz = [];
        const finz = [];
        for (let i = 0; i < dz.length; i++) {
          if (dz[i] === 1) {
            debz.push(i);
          } else if (dz[i] === -1) {
            finz.push(i - 1);
          }
        }

        const indz = [];
        for (let i = 0; i < debz.length; i++) {
          indz.push(Math.round((debz[i] + finz[i]) / 2));
        }

        indzer.push(...indz);
      } else {
        indzer.push(...iz);
      }
    } else {
      indzer.push(...iz);
    }
  }

  indzer.sort((a, b) => a - b);
  return indzer;
}

//========================================================================================

function boundaryConditions(indmin, indmax, t, x, z, nbsym) {
  var lx = x.length - 1;
  var end_max = indmax.length - 1;
  var end_min = indmin.length - 1;
  var indminInt = indmin.map(Number);
  var indmaxInt = indmax.map(Number);
  var tmin, tmax, zmin, zmax;

  if (indmin.length + indmax.length < 3) {
    var mode = 0;
    tmin = tmax = zmin = zmax = null;
    return [tmin, tmax, zmin, zmax, mode];
  } else {
    var mode = 1; // the projected signal has inadequate extrema
    return [tmin, tmax, zmin, zmax, mode];
  }
  if (indmax[0] < indmin[0]) {
    if (x[0] > x[indmin[0]]) {
      var lmax = indmax.slice(1, Math.min(end_max + 1, nbsym + 1)).reverse();
      var lmin = indmin.slice(0, Math.min(end_min + 1, nbsym)).reverse();
      var lsym = indmax[0];
    } else {
      var lmax = indmax.slice(0, Math.min(end_max + 1, nbsym)).reverse();
      var lmin = indmin
        .slice(0, Math.min(end_min + 1, nbsym - 1))
        .reverse()
        .concat([0]);
      var lsym = 0;
    }
  } else {
    if (x[0] < x[indmax[0]]) {
      var lmax = indmax.slice(0, Math.min(end_max + 1, nbsym)).reverse();
      var lmin = indmin.slice(1, Math.min(end_min + 1, nbsym + 1)).reverse();
      var lsym = indmin[0];
    } else {
      var lmax = indmax
        .slice(0, Math.min(end_max + 1, nbsym - 1))
        .reverse()
        .concat([0]);
      var lmin = indmin.slice(0, Math.min(end_min + 1, nbsym)).reverse();
      var lsym = 0;
    }
  }
  if (indmax[indmax.length - 1] < indmin[indmin.length - 1]) {
    if (x[x.length - 1] < x[indmax[indmax.length - 1]]) {
      var rmax = indmax.slice(Math.max(end_max - nbsym + 1, 0)).reverse();
      var rmin = indmin.slice(Math.max(end_min - nbsym, 0), -1).reverse();
      var rsym = indmin[indmin.length - 1];
    } else {
      var rmax = [lx]
        .concat(indmax.slice(Math.max(end_max - nbsym + 2, 0)))
        .reverse();
      var rmin = indmin.slice(Math.max(end_min - nbsym + 1, 0)).reverse();
      var rsym = lx;
    }
  } else {
    if (x[x.length - 1] > x[indmin[indmin.length - 1]]) {
      var rmax = indmax.slice(Math.max(end_max - nbsym, 0), -1).reverse();
      var rmin = indmin.slice(Math.max(end_min - nbsym + 1, 0)).reverse();
      var rsym = indmax[indmax.length - 1];
    } else {
      var rmax = indmax.slice(Math.max(end_max - nbsym + 1, 0)).reverse();
      var rmin = [lx]
        .concat(indmin.slice(Math.max(end_min - nbsym + 2, 0)))
        .reverse();
      var rsym = lx;
    }
  }
  var tlmin = Array.from(
    { length: lmin.length },
    (_, i) => 2 * t[lsym] - t[lmin[i]]
  );
  var tlmax = Array.from(
    { length: lmax.length },
    (_, i) => 2 * t[lsym] - t[lmax[i]]
  );
  var trmin = Array.from(
    { length: rmin.length },
    (_, i) => 2 * t[rsym] - t[rmin[i]]
  );
  var trmax = Array.from(
    { length: rmax.length },
    (_, i) => 2 * t[rsym] - t[rmax[i]]
  );

  // in case symmetrized parts do not extend enough
  if (tlmin[0] > t[0] || tlmax[0] > t[0]) {
    if (lsym == indmax[0]) {
      lmax = indmax.slice(0, Math.min(end_max + 1, nbsym)).reverse();
    } else {
      lmin = indmin.slice(0, Math.min(end_min + 1, nbsym)).reverse();
    }
    if (lsym == 1) {
      console.error("bug");
      // or throw new Error('bug');
    }
    lsym = 0;
    tlmin = Array.from(
      { length: lmin.length },
      (_, i) => 2 * t[lsym] - t[lmin[i]]
    );
    tlmax = Array.from(
      { length: lmax.length },
      (_, i) => 2 * t[lsym] - t[lmax[i]]
    );
  }

  if (trmin[trmin.length - 1] < t[lx] || trmax[trmax.length - 1] < t[lx]) {
    if (rsym == indmax[indmax.length - 1]) {
      rmax = indmax.slice(Math.max(end_max - nbsym + 1, 0)).reverse();
    } else {
      rmin = indmin.slice(Math.max(end_min - nbsym + 1, 0)).reverse();
    }
    if (rsym == lx) {
      throw new Error("bug");
    }
    rsym = lx;
    trmin = Array.from(
      { length: rmin.length },
      (_, i) => 2 * t[rsym] - t[rmin[i]]
    );
    trmax = Array.from(
      { length: rmax.length },
      (_, i) => 2 * t[rsym] - t[rmax[i]]
    );
  }

  var zlmax = lmax.map((i) => z[i]);
  var zlmin = lmin.map((i) => z[i]);
  var zrmax = rmax.map((i) => z[i]);
  var zrmin = rmin.map((i) => z[i]);

  var tmin = tlmin.concat(indmin.map((i) => t[i])).concat(trmin);
  var tmax = tlmax.concat(indmax.map((i) => t[i])).concat(trmax);
  var zmin = zlmin.concat(indmin.map((i) => z[i])).concat(zrmin);
  var zmax = zlmax.concat(indmax.map((i) => z[i])).concat(zrmax);

  return [tmin, tmax, zmin, zmax, mode];
}

//========================================================================================
function envelope_mean(m, t, seq, ndir, N, N_dim) {
  const NBSYM = 2;
  let count = 0;

  const env_mean = new Array(t.length);
  const amp = new Array(t.length);
  const nem = new Array(ndir);
  const nzm = new Array(ndir);

  const dir_vec = new Array(N_dim);
  for (let i = 0; i < N_dim; i++) {
    dir_vec[i] = [0];
  }

  for (let it = 0; it < ndir; it++) {
    if (N_dim !== 3) {
      // Multivariate signal (for N_dim ~=3) with hammersley sequence
      // Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
      const b = 2 * seq[it] - 1;

      // Find angles corresponding to the normalised sequence
      const tht = [];
      for (let j = N_dim - 2; j >= 0; j--) {
        const val = Math.sqrt(
          b
            .slice(j + 1)
            .reverse()
            .reduce((acc, cur) => acc + cur ** 2, 0)
        );
        tht.push(Math.atan2(val, b[j]));
      }

      // Find coordinates of unit direction vectors on n-sphere
      dir_vec[N_dim - 1][0] = 1;
      for (let j = N_dim - 2; j >= 0; j--) {
        dir_vec[j][0] = Math.cos(tht[N_dim - 2 - j]) * dir_vec[j + 1][0];
        for (let k = j + 1; k < N_dim - 1; k++) {
          dir_vec[j][0] *= Math.sin(tht[N_dim - 2 - k]);
        }
      }
    } else {
      // Trivariate signal with hammersley sequence
      // Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
      let tt = 2 * seq[it][0] - 1;
      if (tt > 1) {
        tt = 1;
      } else if (tt < -1) {
        tt = -1;
      }

      // Normalize angle from 0 - 2*pi
      const phirad = seq[it][1] * 2 * Math.PI;
      const st = Math.sqrt(1.0 - tt * tt);

      dir_vec[0][0] = st * Math.cos(phirad);
      dir_vec[1][0] = st * Math.sin(phirad);
      dir_vec[2][0] = tt;
    }
    // Projection of input signal on nth (out of total ndir) direction vectors
    const y = math.multiply(m, dir_vec);

    // Calculates the extrema of the projected signal
    const [indmin, indmax] = local_peaks(y); // Assuming local_peaks() is a defined function
    nem[it] = indmin.length + indmax.length;
    const indzer = zero_crossings(y); // Assuming zero_crossings() is a defined function
    nzm[it] = indzer.length;

    const [tmin, tmax, zmin, zmax, mode] = boundary_conditions(
      indmin,
      indmax,
      t,
      y,
      m,
      NBSYM
    ); // Assuming boundary_conditions() is a defined function

    // Calculate multidimensional envelopes using spline interpolation
    // Only done if the number of extrema of the projected signal exceeds 3
    if (mode) {
      const fmin = new CubicSpline(tmin, zmin, "not-a-knot");
      const env_min = fmin.evaluate(t);
      const fmax = new CubicSpline(tmax, zmax, "not-a-knot");
      const env_max = fmax.evaluate(t);
      for (let i = 0; i < t.length; i++) {
        amp[i] +=
          math.sqrt(math.sum(math.dotPow(env_max[i].subtract(env_min[i]), 2))) /
          2;
        env_mean[i] = env_mean[i].add(env_max[i].add(env_min[i])).divide(2);
      }
    } else {
      // if the projected signal has inadequate extrema
      count++;
    }
  }
  if (ndir > count) {
    for (let i = 0; i < env_mean.length; i++) {
      env_mean[i] = env_mean[i].divide(ndir - count);
      amp[i] = amp[i] / (ndir - count);
    }
  } else {
    env_mean = new Array(N);
    for (let i = 0; i < N; i++) {
      env_mean[i] = new Array(N_dim).fill(0);
    }
    amp = new Array(N).fill(0);
    nem = new Array(ndir).fill(0);
  }

  return [env_mean, nem, nzm, amp];
}

//========================================================================================
function stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim) {
  let env_mean, nem, nzm, amp;
  let stp;
  try {
    [env_mean, nem, nzm, amp] = envelope_mean(m, t, seq, ndir, N, N_dim); // Assuming envelope_mean() is a defined function

    const sx = math.sqrt(math.sum(math.dotPow(env_mean, 2), 1));

    if (amp.every((a) => a !== 0)) {
      sx.forEach((s, i) => {
        sx[i] = s / amp[i];
      });
    }

    const meanSxGreaterThanSd = math.mean(sx.map((s) => s > sd));
    const anySxGreaterThanSd2 = sx.some((s) => s > sd2);
    const anyNemGreaterThan2 = nem.some((n) => n > 2);

    if (
      !(meanSxGreaterThanSd > tol || anySxGreaterThanSd2 || !anyNemGreaterThan2)
    ) {
      stp = 1;
    } else {
      stp = 0;
    }
  } catch (error) {
    env_mean = new Array(N);
    for (let i = 0; i < N; i++) {
      env_mean[i] = new Array(N_dim).fill(0);
    }
    stp = 1;
  }

  return [stp, env_mean];
}

//========================================================================================
function fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim) {
  let env_mean, nem, nzm, amp;
  let stp;
  try {
    [env_mean, nem, nzm, amp] = envelope_mean(m, t, seq, ndir, N, N_dim); // Assuming envelope_mean() is a defined function

    const absDiffNzmNem = math.abs(math.subtract(nzm, nem));
    if (absDiffNzmNem.every((diff) => diff > 1)) {
      stp = 0;
      counter = 0;
    } else {
      counter++;
      stp = counter >= stp_cnt ? 1 : 0;
    }
  } catch (error) {
    env_mean = new Array(N);
    for (let i = 0; i < N; i++) {
      env_mean[i] = new Array(N_dim).fill(0);
    }
    stp = 1;
  }

  return [stp, env_mean, counter];
}

//========================================================================================
function peaks(X) {
  const dX = math.sign(math.diff(X.transpose())).transpose();
  const locs_max = math
    .where(
      math.logicalAnd(
        math.subtract(dX.slice(0, -1), 0) > 0,
        math.subtract(dX.slice(1), 0) < 0
      )
    )
    .map((index) => index + 1);
  const pks_max = math.subset(X, math.index(locs_max));

  return [pks_max, locs_max];
}

//========================================================================================
function local_peaks(x) {
  if (math.every(x, (value) => value < 1e-5)) {
    x = math.zeros([1, x.length]);
  }

  const m = x.length - 1;

  // Calculates the extrema of the projected signal
  // Difference between subsequent elements
  const dy = math.diff(x.transpose()).transpose();
  const a = math.where(math.notEqual(dy, 0))[0];
  const lm = math
    .where(math.notEqual(math.diff(a), 1))[0]
    .map((index) => index + 1);
  const d = math.subtract(
    math.subset(a, math.index(lm)),
    math.floor(math.divide(math.subset(d, math.index(lm)), 2))
  );
  math.subset(
    a,
    math.index(lm),
    math.subtract(math.subset(a, math.index(lm)), d)
  );
  math.subset(a, math.index(a.length), m);
  const ya = math.subset(x, math.index(a));

  let indmin, indmax;
  if (ya.length > 1) {
    // Maxima
    const [pks_max, loc_max] = peaks(ya);
    // Minima
    const [pks_min, loc_min] = peaks(math.unaryMinus(ya));

    if (pks_min.length > 0) {
      indmin = math.subset(a, math.index(loc_min));
    } else {
      indmin = [];
    }

    if (pks_max.length > 0) {
      indmax = math.subset(a, math.index(loc_max));
    } else {
      indmax = [];
    }
  } else {
    indmin = [];
    indmax = [];
  }

  return [indmin, indmax];
}
//==================================================================================
function stop_emd(r, seq, ndir, N_dim) {
  const ner = math.zeros([ndir, 1]);
  const dir_vec = math.zeros([N_dim, 1]);

  for (let it = 0; it < ndir; it++) {
    if (N_dim !== 3) {
      // Multivariate signal (for N_dim ~=3) with hammersley sequence
      // Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
      const b = math.subtract(math.multiply(2, seq[it]), 1);

      // Find angles corresponding to the normalised sequence
      const tht = math
        .atan2(
          math.sqrt(
            math.flip(
              math.cumsum(
                math.slice(b, math.index(0, -1)).map((val) => math.pow(val, 2))
              )
            )
          ),
          math.slice(b, math.index(0, N_dim - 1))
        )
        .transpose();

      // Find coordinates of unit direction vectors on n-sphere
      math.subset(
        dir_vec,
        math.index([0]),
        math.cumprod(math.concat([1], math.sin(tht)))
      );
      math.subset(
        dir_vec,
        math.index(math.range(0, N_dim - 1)),
        math.multiply(
          math.cos(tht),
          math.subset(dir_vec, math.index(math.range(0, N_dim - 1)))
        )
      );
    } else {
      // Trivariate signal with hammersley sequence
      // Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
      let tt = math.multiply(2, seq[it][0]) - 1;
      if (tt > 1) {
        tt = 1;
      } else if (tt < -1) {
        tt = -1;
      }

      // Normalize angle from 0 - 2*pi
      const phirad = math.multiply(seq[it][1], 2 * math.PI);
      const st = math.sqrt(1.0 - math.square(tt));

      math.subset(
        dir_vec,
        math.index([0]),
        math.multiply(st, math.cos(phirad))
      );
      math.subset(
        dir_vec,
        math.index([1]),
        math.multiply(st, math.sin(phirad))
      );
      math.subset(dir_vec, math.index([2]), tt);
    }

    // Projection of input signal on nth (out of total ndir) direction vectors
    const y = math.multiply(r, dir_vec);

    // Calculates the extrema of the projected signal
    const [indmin, indmax] = local_peaks(y);

    math.subset(ner, math.index(it), math.add(indmin.length, indmax.length));
  }

  // Stops if all projected signals have less than 3 extrema
  const stp = math.every(math.smaller(ner, 3));

  return stp;
}

//==================================================================================
function is_prime(x) {
  if (x === 2) {
    return true;
  } else {
    for (let number = 3; number < x; number++) {
      if (x % number === 0 || x % 2 === 0) {
        return false;
      }
    }
  }
  return true;
}

//==================================================================================
function nth_prime(n) {
  var lst = [2];
  for (var i = 3; i < 104745; i++) {
    if (is_prime(i) === true) {
      lst.push(i);
      if (lst.length === n) {
        return lst;
      }
    }
  }
}
//==================================================================================
function set_value() {
  var args = arguments[0];
  var narg = args.length;
  var q = args[0];

  var ndir, stp_cnt, MAXITERATIONS, sd, sd2, tol;
  var stp_crit, stp_vec, base;

  if (narg === 0) {
    console.log("Not enough input arguments.");
    return;
  } else if (narg > 4) {
    console.log("Too many input arguments.");
    return;
  } else if (narg === 1) {
    ndir = 64; // default
    stp_crit = "stop"; // default
    stp_vec = [0.075, 0.75, 0.075]; // default
    sd = stp_vec[0];
    sd2 = stp_vec[1];
    tol = stp_vec[2];
  } else if (narg === 2) {
    ndir = args[1];
    stp_crit = "stop"; // default
    stp_vec = [0.075, 0.75, 0.075]; // default
    sd = stp_vec[0];
    sd2 = stp_vec[1];
    tol = stp_vec[2];
  } else if (narg === 3) {
    if (args[1] !== null) {
      ndir = args[1];
    } else {
      ndir = 64; // default
    }
    stp_crit = args[2];
    if (stp_crit === "stop") {
      stp_vec = [0.075, 0.75, 0.075]; // default
      sd = stp_vec[0];
      sd2 = stp_vec[1];
      tol = stp_vec[2];
    } else if (stp_crit === "fix_h") {
      stp_cnt = 2; // default
    }
  } else if (narg === 4) {
    if (args[1] !== null) {
      ndir = args[1];
    } else {
      ndir = 64; // default
    }
    stp_crit = args[2];
    if (args[2] === "stop") {
      stp_vec = args[3];
      sd = stp_vec[0];
      sd2 = stp_vec[1];
      tol = stp_vec[2];
    } else if (args[2] === "fix_h") {
      stp_cnt = args[3];
    }
  }
  // Rescale input signal if required
  if (q.length === 0) {
    return console.log("emptyDataSet. Data set cannot be empty.");
  }
  if (q.length < q[0].length) {
    q = transpose(q);
  }

  // Dimension of input signal
  var N_dim = q[0].length;
  if (N_dim < 3) {
    return console.log(
      "Function only processes the signal having more than 3."
    );
  }

  // Length of input signal
  var N = q.length;

  // Check validity of Input parameters
  if (!Number.isInteger(ndir) || ndir < 6) {
    return console.log(
      "invalid num_dir. num_dir should be an integer greater than or equal to 6."
    );
  }
  if (
    typeof stp_crit !== "string" ||
    (stp_crit !== "stop" && stp_crit !== "fix_h")
  ) {
    return console.log(
      "invalid stop_criteria. stop_criteria should be either fix_h or stop"
    );
  }
  if (!Array.isArray(stp_vec) || stp_vec.some((x) => typeof x !== "number")) {
    return console.log(
      "invalid stop_vector. stop_vector should be a list with three elements e.g. default is [0.75,0.75,0.75]"
    );
  }
  if (stp_cnt !== null) {
    if (!Number.isInteger(stp_cnt) || stp_cnt < 0) {
      return console.log(
        "invalid stop_count. stop_count should be a nonnegative integer."
      );
    }
  }

  // Initializations for Hammersley function
  var base = [-ndir];
  base.push(-ndir);

  // Find the pointset for the given input signal
  var seq;
  if (N_dim === 3) {
    base.push(2);
    seq = new Array(ndir);
    for (var it = 0; it < N_dim - 1; it++) {
      seq[it] = hamm(ndir, base[it]);
    }
  } else {
    // Prime numbers for Hammersley sequence
    var prm = nth_prime(N_dim - 1);
    for (var itr = 1; itr < N_dim; itr++) {
      base.push(prm[itr - 1]);
    }
    seq = new Array(ndir);
    for (var it = 0; it < N_dim; it++) {
      seq[it] = hamm(ndir, base[it]);
    }
  }

  // Define t
  var t = [];
  for (var i = 1; i <= N; i++) {
    t.push(i);
  }

  // Counter
  var nbit = 0;
  var MAXITERATIONS = 1000; // default

  return [
    q,
    seq,
    t,
    ndir,
    N_dim,
    N,
    sd,
    sd2,
    tol,
    nbit,
    MAXITERATIONS,
    stp_crit,
    stp_cnt,
  ];
}

//==================================================================================
function memd() {
  var args = arguments[0];
  var setValues = set_value(args);
  var x = setValues[0];
  var seq = setValues[1];
  var t = setValues[2];
  var ndir = setValues[3];
  var N_dim = setValues[4];
  var N = setValues[5];
  var sd = setValues[6];
  var sd2 = setValues[7];
  var tol = setValues[8];
  var nbit = setValues[9];
  var MAXITERATIONS = setValues[10];
  var stop_crit = setValues[11];
  var stp_cnt = setValues[12];

  var r = x;
  var n_imf = 1;
  var q = [];

  while (stop_emd(r, seq, ndir, N_dim) === false) {
    // current mode
    var m = r;

    // computation of mean and stopping criterion
    var stop_sift, env_mean, counter;
    if (stop_crit === "stop") {
      var stopResult = stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim);
      stop_sift = stopResult[0];
      env_mean = stopResult[1];
      counter = null;
    } else {
      counter = 0;
      var fixResult = fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim);
      stop_sift = fixResult[0];
      env_mean = fixResult[1];
      counter = fixResult[2];
    }

    // In case the current mode is so small that machine precision can cause
    // spurious extrema to appear
    if (Math.max(...Math.abs(m)) < 1e-10 * Math.max(...Math.abs(x))) {
      if (stop_sift === false) {
        console.warn("emd:warning", "forced stop of EMD : too small amplitude");
      } else {
        console.log("forced stop of EMD : too small amplitude");
      }
      break;
    }

    // sifting loop
    while (stop_sift === false && nbit < MAXITERATIONS) {
      // sifting
      m = subtract(m, env_mean);

      // computation of mean and stopping criterion
      if (stop_crit === "stop") {
        var stopResult = stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim);
        stop_sift = stopResult[0];
        env_mean = stopResult[1];
        counter = null;
      } else {
        var fixResult = fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim);
        stop_sift = fixResult[0];
        env_mean = fixResult[1];
        counter = fixResult[2];
      }

      nbit = nbit + 1;

      if (nbit === MAXITERATIONS - 1 && nbit > 100) {
        console.warn(
          "emd:warning",
          "forced stop of sifting : too many iterations"
        );
      }
    }

    q.push(transpose(m));
    n_imf = n_imf + 1;
    r = subtract(r, m);
    nbit = 0;
  }
  // Stores the residue
  q.push(transpose(r));
//   q = np.array(q);
  // sprintf('Elapsed time: %f\n',toc);

  return q;
}

//==================================================================================
//==================================================================================
//==================================================================================
//==================================================================================
