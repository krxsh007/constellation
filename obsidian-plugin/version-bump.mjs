import { readFileSync, writeFileSync } from "fs";

// Obsidian (and BRAT) read manifest.json / versions.json from the repository
// root, so the copies there are kept in lockstep with the plugin's own.

const targetVersion = process.env.npm_package_version;

let manifest = JSON.parse(readFileSync("manifest.json", "utf8"));
const { minAppVersion } = manifest;
manifest.version = targetVersion;
for (const path of ["manifest.json", "../manifest.json"]) {
	writeFileSync(path, JSON.stringify(manifest, null, "\t") + "\n");
}

let versions = JSON.parse(readFileSync("versions.json", "utf8"));
versions[targetVersion] = minAppVersion;
for (const path of ["versions.json", "../versions.json"]) {
	writeFileSync(path, JSON.stringify(versions, null, "\t") + "\n");
}
