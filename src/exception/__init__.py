import os,sys
class CustomException:
    def __init__(self,error_message:Exception,error_detail:sys):
        self.error_message=CustomException.detailled_error_message(error_message=error_message,
                                                                   error_detail=error_detail)
    @staticmethod
    def detailled_error_message(error_message:Exception,error_detail:sys)->str:
        _,_,exce_tb=error_detail.exc_info()

        exception_block_line_number=exce_tb.tb_frame.f_lineno
        try_block_line_number=exce_tb.tb_lineno
        file_name=exce_tb.tb_frame.f_code.co_filename

        error_message=f""" Error occured while execution of : [{file_name}]
        at try block line numbner : [{try_block_line_number}]
        and Exception block line number :[{exception_block_line_number}]
        error_detail :[{error_message}]
        """       
        return error_message
    

    def __str__(self) -> str:
        return self.error_message
    
    def __repr__(self) -> str:
        return CustomException.__name__.str()