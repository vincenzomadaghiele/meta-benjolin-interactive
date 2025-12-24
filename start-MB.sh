#!/bin/sh
/bin/sh -ec 'cd frontend && npx vite'&
/bin/sh -ec 'cd node_server && node .'&
/bin/sh -ec 'cd python_server && conda run -n benjo python latent_space_class.py'
