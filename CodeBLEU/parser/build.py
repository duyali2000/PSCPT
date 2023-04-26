# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-go-master',
    'tree-sitter-javascript-master',
    'tree-sitter-python-master',
    'tree-sitter-php-master',
    'tree-sitter-java-master',
    'tree-sitter-ruby-master',
    'tree-sitter-c-sharp-master',
  ]
)

