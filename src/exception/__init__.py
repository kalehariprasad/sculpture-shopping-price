import os,sys
class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        self.error_message = CustomException.detailed_error_message(error_message=error_message, error_detail=error_detail)

    @staticmethod
    def detailed_error_message(error_message: str, error_detail: sys) -> str:
        _, _, exce_tb = error_detail.exc_info()

        exception_block_line_number = exce_tb.tb_frame.f_lineno
        try_block_line_number = exce_tb.tb_lineno
        file_name = exce_tb.tb_frame.f_code.co_filename

        error_message = f""" Error occurred while execution of : [{file_name}]
        at try block line number : [{try_block_line_number}]
        and Exception block line number : [{exception_block_line_number}]
        error_detail : [{error_message}]
        """
        return error_message

    def __str__(self) -> str:
        return self.error_message
