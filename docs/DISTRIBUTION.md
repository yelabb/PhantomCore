# Distribution Guide

## Distributing PhantomCore Executables

PhantomCore uses GitHub Releases to distribute pre-built executables for Windows, Linux, and macOS.

## For Users

### Downloading Pre-built Binaries

1. Go to the [Releases page](https://github.com/yelabb/PhantomCore/releases)
2. Download the appropriate archive for your platform:
   - **Windows**: `PhantomCore-vX.X.X-Windows-x64.zip`
   - **Linux**: `PhantomCore-vX.X.X-Linux-x64.tar.gz`
   - **macOS**: `PhantomCore-vX.X.X-macOS-x64.tar.gz`

3. Extract the archive:
   ```bash
   # Linux/macOS
   tar -xzf PhantomCore-vX.X.X-Linux-x64.tar.gz
   
   # Windows
   # Use Windows Explorer or: Expand-Archive PhantomCore-vX.X.X-Windows-x64.zip
   ```

4. The package contains:
   - `bin/` - Executable binaries (examples, benchmarks)
   - `lib/` - Static/shared libraries
   - `include/` - Header files for development
   - `README.md`, `LICENSE`, `CHANGELOG.md`

### Using the Binaries

Add the `bin` directory to your PATH or run directly:

```bash
# Linux/macOS
./bin/realtime_demo
./bin/closed_loop_sim

# Windows
.\bin\realtime_demo.exe
.\bin\closed_loop_sim.exe
```

## For Maintainers

### Creating a New Release

1. **Update version numbers** in:
   - `CMakeLists.txt`
   - `CHANGELOG.md`

2. **Commit changes**:
   ```bash
   git add CMakeLists.txt CHANGELOG.md
   git commit -m "Bump version to X.X.X"
   git push
   ```

3. **Create and push a tag**:
   ```bash
   git tag -a vX.X.X -m "Release version X.X.X"
   git push origin vX.X.X
   ```

4. **Automated build**: The release workflow automatically:
   - Creates a GitHub release
   - Builds for Windows, Linux, and macOS
   - Packages executables with libraries and headers
   - Uploads archives to the release

5. **Verify the release**: Check the [Releases page](https://github.com/yelabb/PhantomCore/releases) and download/test the binaries.

### Testing Pre-release Builds

CI builds upload artifacts for every push:

1. Go to [Actions](https://github.com/yelabb/PhantomCore/actions)
2. Click on a workflow run
3. Download artifacts under "Artifacts" section
4. Artifacts are kept for 30 days

### Manual Release (if needed)

If the automated workflow fails, you can manually create a release:

1. Build locally:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DPHANTOMCORE_BUILD_EXAMPLES=ON
   cmake --build build --config Release --parallel
   ```

2. Package manually following the structure in `.github/workflows/release.yml`

3. Create release on GitHub and upload archives

## Package Contents

Each release archive includes:

```
PhantomCore-vX.X.X-Platform/
├── bin/
│   ├── closed_loop_sim[.exe]
│   ├── latency_benchmark[.exe]
│   ├── realtime_demo[.exe]
│   ├── spike_visualizer[.exe]
│   └── phantomcore_benchmarks[.exe]
├── lib/
│   └── libphantomcore.{a,lib,so,dylib}
├── include/
│   ├── phantomcore.hpp
│   └── phantomcore/
│       └── [all header files]
├── README.md
├── LICENSE
└── CHANGELOG.md
```

## Best Practices

### DO:
- ✅ Distribute via GitHub Releases
- ✅ Build for multiple platforms
- ✅ Include documentation and license
- ✅ Use semantic versioning (vX.Y.Z)
- ✅ Sign releases (optional but recommended)
- ✅ Provide checksums for verification

### DON'T:
- ❌ Commit binaries to the repository
- ❌ Use GitHub Artifacts as permanent distribution
- ❌ Forget to update CHANGELOG.md
- ❌ Release without testing the build

## Troubleshooting

### Release workflow failed
- Check the Actions logs for build errors
- Ensure all tests pass on the target platform
- Verify CMake configuration is correct

### Missing executables in package
- Check if examples/benchmarks are enabled in CMake
- Verify the packaging scripts in the workflow
- Ensure executables are built in the correct directory

### Users report missing dependencies
- Document required system libraries in README
- Consider static linking for standalone binaries
- Provide installation instructions for dependencies

## Security

- Never distribute debug builds with symbols in releases
- Use Release or RelWithDebInfo build types
- Consider code signing for Windows/macOS
- Provide SHA256 checksums for verification

## Additional Resources

- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [Semantic Versioning](https://semver.org/)
- [CMake Install Documentation](https://cmake.org/cmake/help/latest/command/install.html)
