if(NOT DEFINED NAME)
    set(NAME noorRayOptixIr)
endif()

if(NOT EXISTS "${BIN2C}")
    message(FATAL_ERROR "CUDA bin2c was not found at ${BIN2C}")
endif()

execute_process(
    COMMAND "${BIN2C}" --const --static --length --name "${NAME}"
            --type char "${INPUT}"
    OUTPUT_FILE "${OUTPUT}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "bin2c failed with exit code ${result}")
endif()

file(READ "${OUTPUT}" embedded)
file(WRITE "${OUTPUT}" "#pragma once\n${embedded}")
