# Writes slangc's preprocessor output (-E), which it only prints to stdout, to
# OUTPUT. Run with: cmake -DSLANGC=... -DINPUT=... -DOUTPUT=... -P PreprocessSlang.cmake
execute_process(COMMAND "${SLANGC}" -E "${INPUT}"
    OUTPUT_FILE "${OUTPUT}"
    COMMAND_ERROR_IS_FATAL ANY)
