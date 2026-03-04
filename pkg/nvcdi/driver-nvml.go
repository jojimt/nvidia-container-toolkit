/**
# Copyright (c) NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
**/

package nvcdi

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/NVIDIA/nvidia-container-toolkit/internal/discover"
	"github.com/NVIDIA/nvidia-container-toolkit/internal/logger"
	"github.com/NVIDIA/nvidia-container-toolkit/pkg/lookup"
)

// NewDriverDiscoverer creates a discoverer for the libraries and binaries associated with a driver installation.
// The supplied NVML Library is used to query the expected driver version.
func (l *nvmllib) NewDriverDiscoverer() (discover.Discover, error) {
	return (*nvcdilib)(l).newDriverVersionDiscoverer()
}

func (l *nvcdilib) newDriverVersionDiscoverer() (discover.Discover, error) {
	version, err := l.driver.Version()
	if err != nil || version == "" || version == "*.*" {
		return nil, fmt.Errorf("failed to determine driver version (%q): %w", version, err)
	}

	libcudasoParentDirPath, err := l.driver.GetDriverLibDirectory()
	if err != nil {
		return nil, fmt.Errorf("failed to get libcuda.so parent path: %w", err)
	}

	libraries, err := l.NewDriverLibraryDiscoverer(version, libcudasoParentDirPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create discoverer for driver libraries: %v", err)
	}

	ipcs, err := discover.NewIPCDiscoverer(l.logger, l.driver.Root)
	if err != nil {
		return nil, fmt.Errorf("failed to create discoverer for IPC sockets: %v", err)
	}

	firmwares, err := l.newDriverFirmwareDiscoverer(version)
	if err != nil {
		return nil, fmt.Errorf("failed to create discoverer for GSP firmware: %v", err)
	}

	binaries := l.newDriverBinariesDiscoverer()

	d := discover.Merge(
		libraries,
		ipcs,
		firmwares,
		binaries,
	)

	return d, nil
}

// NewDriverLibraryDiscoverer creates a discoverer for the libraries associated with the specified driver version.
func (l *nvcdilib) NewDriverLibraryDiscoverer(version string, libcudaSoParentDirPath string) (discover.Discover, error) {
	versionSuffixLibraryMounts, err := l.getVersionSuffixDriverLibraryMounts(version)
	if err != nil {
		return nil, err
	}
	explicitLibraryMounts, err := l.getExplicitDriverLibraryMounts()
	if err != nil {
		return nil, err
	}

	libraries := discover.Merge(
		versionSuffixLibraryMounts,
		explicitLibraryMounts,
	)

	var discoverers []discover.Discover

	driverDotSoSymlinksDiscoverer := discover.WithDriverDotSoSymlinks(
		l.logger,
		libraries,
		// Since we don't only match version suffixes, we now need to match on wildcards.
		"",
		l.hookCreator,
	)
	discoverers = append(discoverers, driverDotSoSymlinksDiscoverer)

	cudaCompatLibHookDiscoverer := discover.NewCUDACompatHookDiscoverer(l.logger, l.hookCreator, &discover.EnableCUDACompatHookOptions{HostDriverVersion: version})
	discoverers = append(discoverers, cudaCompatLibHookDiscoverer)

	updateLDCache, _ := discover.NewLDCacheUpdateHook(l.logger, libraries, l.hookCreator)
	discoverers = append(discoverers, updateLDCache)

	disableDeviceNodeModification := l.hookCreator.Create(DisableDeviceNodeModificationHook)
	discoverers = append(discoverers, disableDeviceNodeModification)

	driverLibDirectory, err := l.driver.GetDriverLibDirectory()
	if err != nil {
		return nil, fmt.Errorf("failed to get libcuda.so parent directory path: %w", err)
	}
	environmentVariable := &discover.EnvVar{
		Name:  "NVIDIA_CTK_LIBCUDA_DIR",
		Value: driverLibDirectory,
	}
	discoverers = append(discoverers, environmentVariable)

	d := discover.Merge(discoverers...)

	return d, nil
}

func (l *nvcdilib) getVersionSuffixDriverLibraryMounts(version string) (discover.Discover, error) {
	versionSuffixLibraryPaths, err := l.getVersionLibs(version)
	if err != nil {
		return nil, fmt.Errorf("failed to get libraries for driver version: %v", err)
	}
	l.logger.Infof("getVersionSuffixDriverLibraryMounts: using %d path(s) for version-suffix library mounts (driver root=%q)", len(versionSuffixLibraryPaths), l.driver.Root)

	mounts := discover.NewMounts(
		l.logger,
		lookup.NewFileLocator(
			lookup.WithLogger(l.logger),
			lookup.WithRoot(l.driver.Root),
		),
		l.driver.Root,
		versionSuffixLibraryPaths,
	)

	return mounts, nil
}

func (l *nvcdilib) getExplicitDriverLibraryMounts() (discover.Discover, error) {
	if !l.featureFlags[FeatureEnableExplicitDriverLibraries] {
		return nil, nil
	}

	// List of explicit libraries to locate
	// TODO(ArangoGutierrez): we should load the version of the libraries from
	// the sandboxutils-filelist or have a way to allow users to specify the
	// libraries to mount from the config file.
	explicitLibraries := []string{
		"libEGL.so",
		"libGL.so",
		"libGLESv1_CM.so",
		"libGLESv2.so",
		"libGLX.so",
		"libGLdispatch.so",
		"libOpenCL.so",
		"libOpenGL.so",
		"libnvidia-api.so",
		"libnvidia-egl-xcb.so",
		"libnvidia-egl-xlib.so",
	}

	// Include "nvidia" subdir so libs in usr/lib/<arch>/nvidia/ are found
	driverLibraryLocator, err := l.driver.DriverLibraryLocator("nvidia")
	if err != nil {
		return nil, fmt.Errorf("failed to get driver library locator: %w", err)
	}
	mounts := discover.NewMounts(
		l.logger,
		driverLibraryLocator,
		l.driver.Root,
		explicitLibraries,
	)

	return mounts, nil

}

func getUTSRelease() (string, error) {
	utsname := &unix.Utsname{}
	if err := unix.Uname(utsname); err != nil {
		return "", err
	}
	return unix.ByteSliceToString(utsname.Release[:]), nil
}

func getFirmwareSearchPaths(logger logger.Interface) ([]string, error) {

	var firmwarePaths []string
	if p := getCustomFirmwareClassPath(logger); p != "" {
		logger.Debugf("using custom firmware class path: %s", p)
		firmwarePaths = append(firmwarePaths, p)
	}

	utsRelease, err := getUTSRelease()
	if err != nil {
		return nil, fmt.Errorf("failed to get UTS_RELEASE: %v", err)
	}

	standardPaths := []string{
		filepath.Join("/lib/firmware/updates/", utsRelease),
		"/lib/firmware/updates/",
		filepath.Join("/lib/firmware/", utsRelease),
		"/lib/firmware/",
	}

	return append(firmwarePaths, standardPaths...), nil
}

// getCustomFirmwareClassPath returns the custom firmware class path if it exists.
func getCustomFirmwareClassPath(logger logger.Interface) string {
	customFirmwareClassPath, err := os.ReadFile("/sys/module/firmware_class/parameters/path")
	if err != nil {
		logger.Warningf("failed to get custom firmware class path: %v", err)
		return ""
	}

	return strings.TrimSpace(string(customFirmwareClassPath))
}

// newDriverFirmwareDiscoverer creates a discoverer for GSP firmware associated with the specified driver version.
func (l *nvcdilib) newDriverFirmwareDiscoverer(version string) (discover.Discover, error) {
	gspFirmwareSearchPaths, err := getFirmwareSearchPaths(l.logger)
	if err != nil {
		return nil, fmt.Errorf("failed to get firmware search paths: %v", err)
	}
	gspFirmwarePaths := filepath.Join("nvidia", version, "gsp*.bin")
	return discover.NewMounts(
		l.logger,
		lookup.NewFileLocator(
			lookup.WithLogger(l.logger),
			lookup.WithRoot(l.driver.Root),
			lookup.WithSearchPaths(gspFirmwareSearchPaths...),
		),
		l.driver.Root,
		[]string{gspFirmwarePaths},
	), nil
}

// newDriverBinariesDiscoverer creates a discoverer for GSP firmware associated with the GPU driver.
func (l *nvcdilib) newDriverBinariesDiscoverer() discover.Discover {
	return discover.NewMounts(
		l.logger,
		lookup.NewExecutableLocator(l.logger, l.driver.Root),
		l.driver.Root,
		[]string{
			"nvidia-smi",              /* System management interface */
			"nvidia-debugdump",        /* GPU coredump utility */
			"nvidia-persistenced",     /* Persistence mode utility */
			"nvidia-cuda-mps-control", /* Multi process service CLI */
			"nvidia-cuda-mps-server",  /* Multi process service server */
			"nvidia-imex",             /* NVIDIA IMEX Daemon */
			"nvidia-imex-ctl",         /* NVIDIA IMEX control */
		},
	)
}

// getVersionLibs returns libraries under the driver lib directory that should be
// mounted. It includes:
//  1. All files matching *.so.<version> (driver version, e.g. 590.48.01)
//  2. All files matching *.so.* (any version suffix) so that libs with
//     library-specific versions (e.g. libnvidia-egl-gbm.so.1.1.3) are included.
//
// The driver lib directory is walked recursively so libs in any subdir are found.
func (l *nvcdilib) getVersionLibs(version string) ([]string, error) {
	l.logger.Infof("Using driver version %v", version)

	driverLibDir, err := l.driver.GetDriverLibDirectory()
	if err != nil {
		return nil, fmt.Errorf("failed to get driver lib directory: %w", err)
	}

	root := l.driver.Root
	if root == "" {
		root = "/"
	}
	root = filepath.Clean(root)
	// Avoid doubling the root: GetDriverLibDirectory() can return a path that is
	// relative to "/" but already contains the root path (e.g. "kata-containers/.../rootfs-.../usr/lib/...")
	// when the library locator uses root "/". Strip that prefix so Join(root, rel) is correct.
	relLibDir := filepath.Clean(driverLibDir)
	if !filepath.IsAbs(relLibDir) && root != "" && root != "/" {
		rootRel := strings.TrimPrefix(root, string(filepath.Separator))
		if rootRel != "" && strings.HasPrefix(relLibDir, rootRel) {
			relLibDir = strings.TrimPrefix(relLibDir, rootRel)
			relLibDir = strings.TrimPrefix(relLibDir, string(filepath.Separator))
		}
	}
	absLibDir := filepath.Join(root, relLibDir)
	if _, err := os.Stat(absLibDir); err != nil {
		l.logger.Warningf("getVersionLibs: driver lib dir does not exist or not accessible: %q (root=%q driverLibDir=%q): %v", absLibDir, root, driverLibDir, err)
		return nil, fmt.Errorf("driver lib dir %q: %w", absLibDir, err)
	}
	l.logger.Infof("getVersionLibs: walking driver lib dir %q (root=%q)", absLibDir, root)
	driverVersionPattern := "*.so." + version
	anyVersionPattern := "*.so.*"
	var libs []string
	err = filepath.WalkDir(absLibDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		base := filepath.Base(path)
		// Match *.so.<driverVersion> or any *.so.<anything> (versioned shared lib)
		matched, _ := filepath.Match(driverVersionPattern, base)
		if !matched {
			matched, _ = filepath.Match(anyVersionPattern, base)
		}
		if !matched {
			return nil
		}
		rel := strings.TrimPrefix(path, root)
		rel = strings.TrimPrefix(rel, string(filepath.Separator))
		if rel == "" {
			return nil
		}
		libs = append(libs, rel)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to walk driver lib dir %q: %w", absLibDir, err)
	}

	seen := make(map[string]bool)
	var out []string
	for _, p := range libs {
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	l.logger.Infof("getVersionLibs: found %d versioned lib(s) under %q (after dedupe)", len(out), absLibDir)
	for i, p := range out {
		if i < 5 || i >= len(out)-2 {
			l.logger.Debugf("getVersionLibs: [%d] %s", i, p)
		} else if i == 5 {
			l.logger.Debugf("getVersionLibs: ... (%d more)", len(out)-7)
		}
	}
	return out, nil
}
